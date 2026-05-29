package main

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
)

type Handlers struct {
	db *Database
}

func NewHandlers(db *Database) *Handlers {
	return &Handlers{db: db}
}

// Enable CORS middleware helper
func (h *Handlers) enableCORS(w http.ResponseWriter, r *http.Request) bool {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
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
