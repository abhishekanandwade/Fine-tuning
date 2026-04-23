package domain

import "time"

type User struct {
	ID        int64     `json:"id" db:"id"`
	FirstName string    `json:"first_name" db:"first_name"`
	LastName  string    `json:"last_name" db:"last_name"`
	Age       int       `json:"age" db:"age"`
	City      string    `json:"city" db:"city"`
	Email     string    `json:"email" db:"email"`
	CreatedAt time.Time `json:"created_at" db:"created_at"`
	UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
}

// BAD: underscore naming instead of MixedCaps — NAM-001 violation
var user_cache_ttl = 300

// BAD: acronym not all caps — NAM-002 violation
type HttpConfig struct {
	BaseUrl    string
	TimeoutMs  int
	MaxRetries int
}

// BAD: function uses underscores — NAM-001 violation
func get_user_display_name(u User) string {
	return u.FirstName + " " + u.LastName
}

// ProcessPayment uses panic for recoverable error — EH-003 violation
func ProcessPayment(amount float64) {
	if amount <= 0 {
		panic("invalid payment amount")
	}
}
