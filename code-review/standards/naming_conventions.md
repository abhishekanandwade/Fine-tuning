# Naming Conventions

## Overview
Go has strong conventions for naming that are enforced by the community and tooling.
Following these conventions makes code instantly readable to any Go developer.

## Rules

### NAM-001: Use MixedCaps, Not Underscores (LOW)
Go uses MixedCaps (camelCase for unexported, PascalCase for exported) for all names.
Underscores are not idiomatic except in test function names and cgo.

**Bad:**
```go
var user_name string
func get_user_by_id() {}
const max_retry_count = 3
```

**Good:**
```go
var userName string
func getUserByID() {}
const maxRetryCount = 3
```

### NAM-002: Acronyms Should Be All Caps (LOW)
Common acronyms like URL, HTTP, ID, API, JSON should be all uppercase or all lowercase.

**Bad:**
```go
type HttpClient struct {}
var usrId int
func parseJson() {}
type ApiResponse struct {}
```

**Good:**
```go
type HTTPClient struct {}
var userID int
func parseJSON() {}
type APIResponse struct {}
```

### NAM-003: Receiver Names Should Be Short (LOW)
Method receivers should be one or two letters, typically the first letter(s) of the type.
Never use `self` or `this`.

**Bad:**
```go
func (this *Server) Start() {}
func (self *UserService) GetUser() {}
func (server *Server) Stop() {}
```

**Good:**
```go
func (s *Server) Start() {}
func (us *UserService) GetUser() {}
func (s *Server) Stop() {}
```

### NAM-004: Interface Names End in -er for Single-Method (LOW)
Single-method interfaces should be named with the method name plus an -er suffix.

**Good:**
```go
type Reader interface { Read(p []byte) (n int, err error) }
type Writer interface { Write(p []byte) (n int, err error) }
type Stringer interface { String() string }
```

### NAM-005: Package Names Are Lowercase, Single Word (MEDIUM)
Package names should be short, concise, lowercase, and single-word. Avoid underscores,
hyphens, or mixedCaps in package names.

**Bad:**
```go
package userService
package user_service
package util  // too generic
```

**Good:**
```go
package user
package auth
package payment
```
