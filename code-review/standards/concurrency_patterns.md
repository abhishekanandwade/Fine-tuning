# Concurrency Patterns

## Overview
Go's concurrency primitives (goroutines, channels, sync package) are powerful but require
discipline to use correctly. Race conditions and goroutine leaks are common in Go code.

## Rules

### CONC-001: Goroutines Must Have a Clear Termination Path (HIGH)
Every goroutine launched must have a clear mechanism to stop: context cancellation,
done channel, or WaitGroup. Goroutine leaks are silent memory/CPU leaks.

**Bad:**
```go
func startWorker() {
    go func() {
        for {
            processItem()  // no way to stop this
        }
    }()
}
```

**Good:**
```go
func startWorker(ctx context.Context) {
    go func() {
        for {
            select {
            case <-ctx.Done():
                return
            default:
                processItem()
            }
        }
    }()
}
```

### CONC-002: Use Directional Channels When Possible (MEDIUM)
Function parameters should use directional channel types to express intent.

**Bad:**
```go
func producer(ch chan int) { ch <- 42 }
func consumer(ch chan int) { v := <-ch }
```

**Good:**
```go
func producer(ch chan<- int) { ch <- 42 }   // send-only
func consumer(ch <-chan int) { v := <-ch }  // receive-only
```

### CONC-003: Protect Shared State with sync.Mutex (HIGH)
Shared mutable state accessed from multiple goroutines must be protected with
sync.Mutex or sync.RWMutex. Use RWMutex when reads far outnumber writes.

**Bad:**
```go
type Cache struct {
    data map[string]string  // unprotected!
}

func (c *Cache) Set(k, v string) { c.data[k] = v }
func (c *Cache) Get(k string) string { return c.data[k] }
```

**Good:**
```go
type Cache struct {
    mu   sync.RWMutex
    data map[string]string
}

func (c *Cache) Set(k, v string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.data[k] = v
}

func (c *Cache) Get(k string) string {
    c.mu.RLock()
    defer c.mu.RUnlock()
    return c.data[k]
}
```

### CONC-004: Use sync.WaitGroup for Fan-Out Patterns (MEDIUM)
When launching multiple goroutines, use sync.WaitGroup to wait for all to complete.

**Good:**
```go
func processAll(ctx context.Context, items []Item) error {
    var wg sync.WaitGroup
    errCh := make(chan error, len(items))

    for _, item := range items {
        wg.Add(1)
        go func(it Item) {
            defer wg.Done()
            if err := process(ctx, it); err != nil {
                errCh <- err
            }
        }(item)
    }

    wg.Wait()
    close(errCh)

    for err := range errCh {
        return err  // return first error
    }
    return nil
}
```
