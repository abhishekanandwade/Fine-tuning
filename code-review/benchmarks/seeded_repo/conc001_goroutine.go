package seeded

import "time"

// Seeded violation: CONC-001 (infinite goroutine with no ctx.Done / cancellation)
func StartBackgroundWorker() {
	go func() {
		for {
			doWork()
			time.Sleep(1 * time.Second)
		}
	}()
}

func doWork() {}
