# Security Guidelines

## Overview
Security vulnerabilities in Go services can have severe consequences. These rules cover
the most common security issues found in Go codebases.

## Rules

### SEC-001: No Hardcoded Secrets or Credentials (CRITICAL)
Never hardcode passwords, API keys, tokens, or secrets in source code. Use environment
variables, secret managers (Vault, AWS Secrets Manager), or config files excluded from VCS.

**Bad:**
```go
const apiKey = "sk-abc123secretkey"
var dbPassword = "admin123"
```

**Good:**
```go
apiKey := os.Getenv("API_KEY")
if apiKey == "" {
    return fmt.Errorf("API_KEY environment variable not set")
}
```

### SEC-002: Use Parameterized Queries for SQL (CRITICAL)
Never construct SQL queries with string concatenation or fmt.Sprintf. Always use
parameterized queries to prevent SQL injection.

**Bad:**
```go
query := fmt.Sprintf("SELECT * FROM users WHERE id = '%s'", userID)
rows, err := db.Query(query)
```

**Good:**
```go
rows, err := db.QueryContext(ctx, "SELECT * FROM users WHERE id = $1", userID)
```

### SEC-003: Validate and Sanitize All External Input (HIGH)
Never trust input from users, APIs, or external systems. Validate types, ranges,
lengths, and formats before processing.

**Good:**
```go
func handleUserAge(ageStr string) (int, error) {
    age, err := strconv.Atoi(ageStr)
    if err != nil {
        return 0, fmt.Errorf("invalid age format: %w", err)
    }
    if age < 0 || age > 150 {
        return 0, fmt.Errorf("age out of valid range: %d", age)
    }
    return age, nil
}
```

### SEC-004: Use TLS for All Network Communication (HIGH)
All HTTP clients and servers in production must use TLS. Never disable TLS verification
in production code.

**Bad:**
```go
tr := &http.Transport{
    TLSClientConfig: &tls.Config{InsecureSkipVerify: true},  // NEVER in production
}
```

**Good:**
```go
tr := &http.Transport{
    TLSClientConfig: &tls.Config{
        MinVersion: tls.VersionTLS12,
    },
}
```

### SEC-005: Set Timeouts on HTTP Servers and Clients (HIGH)
Always set read, write, and idle timeouts on HTTP servers and clients to prevent
slowloris attacks and resource exhaustion.

**Good:**
```go
srv := &http.Server{
    Addr:         ":8080",
    ReadTimeout:  5 * time.Second,
    WriteTimeout: 10 * time.Second,
    IdleTimeout:  120 * time.Second,
}
```
