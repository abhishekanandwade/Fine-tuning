package seeded

import (
	"database/sql"
	"fmt"
)

// Seeded violation: SEC-002 (SQL string concatenation)
func LookupUser(db *sql.DB, id string) (*sql.Row, error) {
	query := "SELECT * FROM users WHERE id = '" + id + "'"
	return db.QueryRow(query), nil
}

func _unused() { fmt.Println("stub") }
