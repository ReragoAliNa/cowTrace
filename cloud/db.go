package main

import (
	"encoding/json"
	"os"
	"sort"
	"sync"
	"time"
)

type CowRecord struct {
	RunID      string    `json:"run_id,omitempty"`
	VideoName  string    `json:"video_name,omitempty"`
	FrameIndex int       `json:"frame_index"`
	CattleID   int       `json:"cattle_id"`
	Bbox       []int     `json:"bbox"` // [x1, y1, x2, y2]
	Status     string    `json:"status"`
	Timestamp  time.Time `json:"timestamp"`
}

type CowStatus struct {
	CowRecord
	Statuses     []string       `json:"statuses"`
	StatusCounts map[string]int `json:"status_counts"`
	FirstFrame   int            `json:"first_frame"`
	LastFrame    int            `json:"last_frame"`
}

type Database struct {
	mu       sync.RWMutex
	filePath string
	Records  []CowRecord `json:"records"`
}

func NewDatabase(filePath string) *Database {
	db := &Database{
		filePath: filePath,
		Records:  make([]CowRecord, 0),
	}
	db.Load()
	return db
}

func (db *Database) AddRecords(records []CowRecord) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	// Add timestamp to new records if empty
	now := time.Now()
	for i := range records {
		if records[i].Timestamp.IsZero() {
			records[i].Timestamp = now
		}
	}

	db.Records = append(db.Records, records...)
	return db.save()
}

func (db *Database) GetLatestStatus() []CowStatus {
	db.mu.RLock()
	defer db.mu.RUnlock()

	records := db.latestRunRecordsLocked()
	latestMap := make(map[int]CowRecord)
	for _, rec := range records {
		existing, found := latestMap[rec.CattleID]
		if !found || rec.Timestamp.After(existing.Timestamp) || (rec.Timestamp.Equal(existing.Timestamp) && rec.FrameIndex > existing.FrameIndex) {
			latestMap[rec.CattleID] = rec
		}
	}

	// Keep short-lived behavior events visible on the dashboard instead of
	// immediately replacing them with a later Standing classification.
	const behaviorHoldFrames = 60
	for _, rec := range records {
		if !isPriorityBehavior(rec.Status) {
			continue
		}

		latest, found := latestMap[rec.CattleID]
		if !found {
			latestMap[rec.CattleID] = rec
			continue
		}

		sameRun := rec.Timestamp.After(latest.Timestamp.Add(-5*time.Minute)) && rec.Timestamp.Before(latest.Timestamp.Add(5*time.Minute))
		frameDistance := latest.FrameIndex - rec.FrameIndex
		if sameRun && frameDistance >= 0 && frameDistance <= behaviorHoldFrames && behaviorPriority(rec.Status) < behaviorPriority(latest.Status) {
			latestMap[rec.CattleID] = rec
		}
	}

	result := make([]CowStatus, 0, len(latestMap))
	for _, rec := range latestMap {
		statusCounts := make(map[string]int)
		firstFrame := rec.FrameIndex
		lastFrame := rec.FrameIndex
		for _, history := range records {
			if history.CattleID != rec.CattleID {
				continue
			}
			statusCounts[history.Status]++
			if history.FrameIndex < firstFrame {
				firstFrame = history.FrameIndex
			}
			if history.FrameIndex > lastFrame {
				lastFrame = history.FrameIndex
			}
		}

		statuses := make([]string, 0, len(statusCounts))
		for status := range statusCounts {
			statuses = append(statuses, status)
		}
		sort.Slice(statuses, func(i, j int) bool {
			return behaviorPriority(statuses[i]) < behaviorPriority(statuses[j])
		})

		result = append(result, CowStatus{
			CowRecord:    rec,
			Statuses:     statuses,
			StatusCounts: statusCounts,
			FirstFrame:   firstFrame,
			LastFrame:    lastFrame,
		})
	}

	sort.Slice(result, func(i, j int) bool {
		return result[i].CattleID < result[j].CattleID
	})
	return result
}

func (db *Database) latestRunRecordsLocked() []CowRecord {
	if len(db.Records) == 0 {
		return nil
	}

	latest := db.Records[0]
	for _, rec := range db.Records[1:] {
		if rec.Timestamp.After(latest.Timestamp) || (rec.Timestamp.Equal(latest.Timestamp) && rec.FrameIndex > latest.FrameIndex) {
			latest = rec
		}
	}

	records := make([]CowRecord, 0)
	if latest.RunID != "" {
		for _, rec := range db.Records {
			if rec.RunID == latest.RunID {
				records = append(records, rec)
			}
		}
	} else {
		start := latest.Timestamp.Add(-5 * time.Minute)
		end := latest.Timestamp.Add(5 * time.Minute)
		for _, rec := range db.Records {
			if rec.Timestamp.Before(start) || rec.Timestamp.After(end) {
				continue
			}
			records = append(records, rec)
		}
	}

	sort.Slice(records, func(i, j int) bool {
		if records[i].FrameIndex == records[j].FrameIndex {
			return records[i].CattleID < records[j].CattleID
		}
		return records[i].FrameIndex < records[j].FrameIndex
	})
	return records
}

func isPriorityBehavior(status string) bool {
	return status == "Estrus" || status == "Grazing" || status == "Lying"
}

func behaviorPriority(status string) int {
	switch status {
	case "Estrus":
		return 0
	case "Grazing":
		return 1
	case "Lying":
		return 2
	default:
		return 3
	}
}

func (db *Database) GetHistory(cattleID int) []CowRecord {
	db.mu.RLock()
	defer db.mu.RUnlock()

	records := db.latestRunRecordsLocked()
	history := make([]CowRecord, 0)
	for _, rec := range records {
		if rec.CattleID == cattleID {
			history = append(history, rec)
		}
	}

	sort.Slice(history, func(i, j int) bool {
		return history[i].FrameIndex < history[j].FrameIndex
	})
	return history
}

func (db *Database) save() error {
	data, err := json.MarshalIndent(db, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(db.filePath, data, 0644)
}

func (db *Database) Load() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if _, err := os.Stat(db.filePath); os.IsNotExist(err) {
		return nil
	}

	data, err := os.ReadFile(db.filePath)
	if err != nil {
		return err
	}

	return json.Unmarshal(data, db)
}
