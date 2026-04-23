package seeded

import (
	"context"
	"fmt"
	"log/slog"
)

// Negative test: NO violations expected.
func FetchUser(ctx context.Context, userID int) (string, error) {
	if userID <= 0 {
		return "", fmt.Errorf("FetchUser: invalid userID=%d", userID)
	}
	slog.InfoContext(ctx, "fetching user", "user_id", userID)
	return "alice", nil
}
