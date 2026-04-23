package seeded

// Seeded violation: SEC-001 (hardcoded credential)
const apiKey = "sk-abc123secretkey"

func GetAPIKey() string {
	return apiKey
}
