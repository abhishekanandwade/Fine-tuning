package seeded

import (
	"encoding/json"
	"os"
)

// Seeded violation: EH-002 (ignored error return with _)
func WriteConfig(path string, cfg any) {
	data, _ := json.Marshal(cfg)
	os.WriteFile(path, data, 0644)
}
