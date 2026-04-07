# Error Handling Standards

## Overview
Error handling is one of the most critical aspects of Go code quality. Go uses explicit 
error returns instead of exceptions, which makes error handling visible but also means 
it must be done consistently and thoroughly.

## Rules

### EH-001: Always Wrap Errors with Context (HIGH)
Bare error returns lose the call chain context. Use `fmt.Errorf` with `%w` to wrap 
errors so the full trace is preserved for debugging.

**Bad:**
```go
func getUser(id int) (User, error) {
    u, err := db.Query(id)
    if err != nil {
        return User{}, err  // loses context
    }
    return u, nil
}
```

**Good:**
```go
func getUser(id int) (User, error) {
    u, err := db.Query(id)
    if err != nil {
        return User{}, fmt.Errorf("getUser id=%d: %w", id, err)
    }
    return u, nil
}
```

### EH-002: Never Ignore Errors (HIGH)
Every error return must be checked. Ignoring errors with `_` leads to silent failures 
that are extremely hard to debug in production.

**Bad:**
```go
data, _ := json.Marshal(config)
os.WriteFile(path, data, 0644)
```

**Good:**
```go
data, err := json.Marshal(config)
if err != nil {
    return fmt.Errorf("marshal config: %w", err)
}
if err := os.WriteFile(path, data, 0644); err != nil {
    return fmt.Errorf("write config to %s: %w", path, err)
}
```

### EH-003: No panic in Library or Service Code (HIGH)
`panic` should never be used for recoverable error conditions in service or library code. 
It crashes the entire process and cannot be handled gracefully by callers.

**Acceptable uses of panic:**
- Program initialization that truly cannot proceed (missing critical config at startup)
- Programmer errors that should never happen (unreachable code)

**Bad:**
```go
func processPayment(amount float64) {
    if amount <= 0 {
        panic("invalid amount")
    }
}
```

**Good:**
```go
func processPayment(amount float64) error {
    if amount <= 0 {
        return fmt.Errorf("processPayment: amount must be positive, got %v", amount)
    }
    return nil
}
```

### EH-004: Use Sentinel Errors for Expected Conditions (MEDIUM)
Define sentinel errors for conditions callers need to check. Use `errors.Is()` and 
`errors.As()` for error comparison.

**Good:**
```go
var ErrNotFound = errors.New("user not found")
var ErrDuplicate = errors.New("user already exists")

func GetUser(id int) (User, error) {
    u, err := db.FindByID(id)
    if err != nil {
        if errors.Is(err, sql.ErrNoRows) {
            return User{}, ErrNotFound
        }
        return User{}, fmt.Errorf("GetUser id=%d: %w", id, err)
    }
    return u, nil
}
```
