package main

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type Handlers struct {
	db *Database
}

type VideoInfo struct {
	Name         string  `json:"name"`
	Path         string  `json:"path"`
	FallbackPath string  `json:"fallback_path"`
	Codec        string  `json:"codec"`
	Mtime        float64 `json:"mtime"`
}

func NewHandlers(db *Database) *Handlers {
	return &Handlers{db: db}
}

// Enable CORS middleware helper
func (h *Handlers) enableCORS(w http.ResponseWriter, r *http.Request) bool {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Cache-Control", "no-store")
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return true
	}
	return false
}

func (h *Handlers) UploadHandler(w http.ResponseWriter, r *http.Request) {
	if h.enableCORS(w, r) {
		return
	}

	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var records []CowRecord
	err := json.NewDecoder(r.Body).Decode(&records)
	if err != nil {
		http.Error(w, "Bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	err = h.db.AddRecords(records)
	if err != nil {
		http.Error(w, "Internal server error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"message": "Data uploaded successfully"})
}

func (h *Handlers) GetCowsHandler(w http.ResponseWriter, r *http.Request) {
	if h.enableCORS(w, r) {
		return
	}

	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	latest := h.db.GetLatestStatus()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(latest)
}

func (h *Handlers) GetVideosHandler(w http.ResponseWriter, r *http.Request) {
	if h.enableCORS(w, r) {
		return
	}

	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	entries := make([]VideoInfo, 0)
	outputsDir := filepath.Join("..", "outputs")
	items, err := os.ReadDir(outputsDir)
	if err == nil {
		for _, item := range items {
			if !item.IsDir() {
				continue
			}

			name := item.Name()
			webmPath := filepath.Join(outputsDir, name, "output_tracked.webm")
			mp4Path := filepath.Join(outputsDir, name, "output_tracked.mp4")
			chosenPath := webmPath
			codec := "WEBM/VP8"
			info, statErr := os.Stat(chosenPath)
			if statErr != nil {
				chosenPath = mp4Path
				codec = "MP4"
				info, statErr = os.Stat(chosenPath)
			}
			if statErr != nil {
				continue
			}

			fallbackPath := mp4Path
			if _, statErr := os.Stat(fallbackPath); statErr != nil {
				fallbackPath = chosenPath
			}

			entries = append(entries, VideoInfo{
				Name:         name,
				Path:         "/outputs/" + name + "/" + filepath.Base(chosenPath),
				FallbackPath: "/outputs/" + name + "/" + filepath.Base(fallbackPath),
				Codec:        codec,
				Mtime:        float64(info.ModTime().Unix()),
			})
		}
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Mtime > entries[j].Mtime
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(entries)
}

func (h *Handlers) GetCowHistoryHandler(w http.ResponseWriter, r *http.Request) {
	if h.enableCORS(w, r) {
		return
	}

	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Simple URL path extraction for ID: /api/cows/{id}/history
	parts := strings.Split(r.URL.Path, "/")
	if len(parts) < 4 {
		http.Error(w, "Bad request: Missing Cow ID", http.StatusBadRequest)
		return
	}

	idStr := parts[3]
	cattleID, err := strconv.Atoi(idStr)
	if err != nil {
		http.Error(w, "Bad request: Invalid Cow ID", http.StatusBadRequest)
		return
	}

	history := h.db.GetHistory(cattleID)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(history)
}
