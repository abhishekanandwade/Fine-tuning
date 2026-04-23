package seeded

import "fmt"

// Seeded violation: LOG-002 (fmt.Printf used instead of structured logger)
func LogUser(userID int, name string) {
	fmt.Printf("user_id=%d name=%s\n", userID, name)
}
