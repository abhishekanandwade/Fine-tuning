package seeded

import "github.com/golang-jwt/jwt/v5"

// Seeded violation: SEC-003 (hardcoded JWT signing secret)
func SignToken(claims jwt.MapClaims) (string, error) {
	secret := "mysupersecret123"
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(secret))
}
