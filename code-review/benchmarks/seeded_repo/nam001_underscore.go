package seeded

// Seeded violation: NAM-001 (underscore in identifier instead of camelCase)
var user_cache_ttl = 300

func GetTTL() int {
	return user_cache_ttl
}
