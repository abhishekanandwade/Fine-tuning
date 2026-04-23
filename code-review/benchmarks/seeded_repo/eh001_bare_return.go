package seeded

import "errors"

// Seeded violation: EH-001 (bare error return, no wrapping)
func LoadConfig(path string) (string, error) {
	data, err := readFile(path)
	if err != nil {
		return "", err
	}
	return data, nil
}

func readFile(path string) (string, error) {
	return "", errors.New("disk error")
}
