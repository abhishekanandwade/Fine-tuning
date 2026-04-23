package seeded

import (
	"context"
	"database/sql"
)

// Seeded violation: CTX-002 (context stored in a struct field)
type Server struct {
	ctx context.Context
	db  *sql.DB
}

func NewServer(ctx context.Context, db *sql.DB) *Server {
	return &Server{ctx: ctx, db: db}
}
