package main

import (
	"embed"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
)

//go:embed web/index.html
var webFiles embed.FS

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8082"
	}

	dbFile := "data.json"
	db := NewDatabase(dbFile)
	handlers := NewHandlers(db)

	// Single entry handler that logs requests and handles clean routing manually
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		log.Printf("[REQ] %s %s", r.Method, r.URL.Path)
		w.Header().Set("Cache-Control", "no-store")

		if r.URL.Path == "/" {
			data, err := webFiles.ReadFile("web/index.html")
			if err != nil {
				http.Error(w, "Internal server error: "+err.Error(), http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Write(data)
			return
		}

		if r.URL.Path == "/api/upload" {
			handlers.UploadHandler(w, r)
			return
		}

		if strings.HasPrefix(r.URL.Path, "/outputs/") {
			filePath := ".." + r.URL.Path
			http.ServeFile(w, r, filePath)
			return
		}

		if r.URL.Path == "/api/cows" {
			handlers.GetCowsHandler(w, r)
			return
		}

		if r.URL.Path == "/api/videos" {
			handlers.GetVideosHandler(w, r)
			return
		}

		if strings.HasPrefix(r.URL.Path, "/api/cows/") && strings.HasSuffix(r.URL.Path, "/history") {
			handlers.GetCowHistoryHandler(w, r)
			return
		}

		http.NotFound(w, r)
	})

	fmt.Printf("==================================================\n")
	fmt.Printf("   Cow Behavior Cloud System Server Starting      \n")
	fmt.Printf("   Listening on http://localhost:%s              \n", port)
	fmt.Printf("==================================================\n")

	err := http.ListenAndServe(":"+port, nil)
	if err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}
