package main

import (
	"encoding/json"
	"os"
	"sync"
	"time"
)

type CowRecord struct {
	FrameIndex int       `json:"frame_index"`
	CattleID   int       `json:"cattle_id"`
	Bbox       []int     `json:"bbox"` // [x1, y1, x2, y2]
	Status     string    `json:"status"`
	Timestamp  time.Time `json:"timestamp"`
}

type Database struct {
	mu      sync.RWMutex
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

func (db *Database) GetLatestStatus() []CowRecord {
	db.mu.RLock()
	defer db.mu.RUnlock()

	latestMap := make(map[int]CowRecord)
	for _, rec := range db.Records {
		existing, found := latestMap[rec.CattleID]
		if !found || rec.Timestamp.After(existing.Timestamp) || (rec.Timestamp.Equal(existing.Timestamp) && rec.FrameIndex > existing.FrameIndex) {
			latestMap[rec.CattleID] = rec
		}
	}

	result := make([]CowRecord, 0, len(latestMap))
	for _, rec := range latestMap {
		result = append(result, rec)
	}
	return result
}

func (db *Database) GetHistory(cattleID int) []CowRecord {
	db.mu.RLock()
	defer db.mu.RUnlock()

	history := make([]CowRecord, 0)
	for _, rec := range db.Records {
		if rec.CattleID == cattleID {
			history = append(history, rec)
		}
	}
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
