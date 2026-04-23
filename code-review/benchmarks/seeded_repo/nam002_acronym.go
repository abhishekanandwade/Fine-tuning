package seeded

// Seeded violation: NAM-002 (acronym not all-caps: HttpConfig, BaseUrl)
type HttpConfig struct {
	BaseUrl string
	ApiKey  string
}

func NewHttpConfig() *HttpConfig {
	return &HttpConfig{}
}
