package handler

import (
	"regexp"

	//"github.com/go-playground/locales/en"
	"github.com/go-playground/validator/v10"
	//en_translations "github.com/go-playground/validator/v10/translations/en"

	validation "gitlab.cept.gov.in/it-2.0-common/api-validation"
)

func NewValidatorService() error {

	err := validation.Create()
	if err != nil {
		return err
	}
	err = validation.RegisterCustomValidation("name", ValidateName, "field %s must consist of letters and spaces only, but received %v")
	if err != nil {
		return nil
	}

	err = validation.RegisterCustomValidation("remarks", validateRemarks, "field %s must consist of letters, spaces, commas and periods only, but received %v")
	if err != nil {
		return nil
	}
	err = validation.RegisterCustomValidation("eightdigitid", validateEightDigits, "field %s must consist of 8 digits, but received %v")
	if err != nil {
		return nil
	}
	return nil
}

func ValidateName(fl validator.FieldLevel) bool {
	// Allows letters and spaces only
	regex := regexp.MustCompile(`^[a-zA-Z\s]+$`)
	return regex.MatchString(fl.Field().String())
}

func validateRemarks(fl validator.FieldLevel) bool {
	reasons := fl.Field().String()
	// Regular expression to allow only alphabets, spaces, commas, and periods.
	regex := `^[a-zA-Z\s,\.]+$`
	matched, _ := regexp.MatchString(regex, reasons)
	return matched
}
func validateEightDigits(fl validator.FieldLevel) bool {
	value := fl.Field().String()
	// Regular expression to match exactly 8 digits.
	regex := `^\d{8}$`
	matched, _ := regexp.MatchString(regex, value)
	return matched
}
