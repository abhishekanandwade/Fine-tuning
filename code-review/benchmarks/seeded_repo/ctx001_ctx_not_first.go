package seeded

import "context"

// Seeded violation: CTX-001 (context not first parameter)
func FetchData(url string, ctx context.Context) ([]byte, error) {
	_ = ctx
	_ = url
	return nil, nil
}
