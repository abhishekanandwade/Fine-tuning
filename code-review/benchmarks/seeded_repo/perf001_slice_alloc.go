package seeded

type User struct {
	Name string
}

// Seeded violation: PERF-001 (slice not pre-allocated when size is known)
func GetNames(users []User) []string {
	var names []string
	for _, u := range users {
		names = append(names, u.Name)
	}
	return names
}
