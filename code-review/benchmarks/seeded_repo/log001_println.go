package seeded

import "fmt"

// Seeded violation: LOG-001 (fmt.Println used instead of structured logger)
func StartServer(port int) {
	fmt.Println("Server starting on port", port)
}
