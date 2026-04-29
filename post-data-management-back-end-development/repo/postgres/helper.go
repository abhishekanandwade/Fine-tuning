package repo

import (
	"reflect"

	"github.com/volatiletech/null/v9"
)

// import (
// 	"database/sql"

// 	sq "github.com/Masterminds/squirrel"
// )

// /**
//  * psql holds a reference to squirrel.StatementBuilderType
//  * which is used to build SQL queries that compatible with PostgreSQL syntax
//  */
// var psql = sq.StatementBuilder.PlaceholderFormat(sq.Dollar)

// // nullString converts a string to sql.NullString for empty string check
// func nullString(value string) sql.NullString {
// 	if value == "" {
// 		return sql.NullString{}
// 	}

// 	return sql.NullString{
// 		String: value,
// 		Valid:  true,
// 	}
// }

// // nullUint64 converts an uint64 to sql.NullInt64 for empty uint64 check
// func nullUint64(value uint64) sql.NullInt64 {
// 	if value == 0 {
// 		return sql.NullInt64{}
// 	}

// 	valueInt64 := int64(value)

// 	return sql.NullInt64{
// 		Int64: valueInt64,
// 		Valid: true,
// 	}
// }

// // nullInt64 converts an int64 to sql.NullInt64 for empty int64 check
// func nullInt64(value int64) sql.NullInt64 {
// 	if value == 0 {
// 		return sql.NullInt64{}
// 	}

// 	return sql.NullInt64{
// 		Int64: value,
// 		Valid: true,
// 	}
// }

// func nullInt(value int) sql.NullInt64 {
// 	if value == 0 {
// 		return sql.NullInt64{}
// 	}
// 	valueInt := int64(value)

// 	return sql.NullInt64{
// 		Int64: valueInt,
// 		Valid: true,
// 	}
// }

// // nullFloat64 converts a float64 to sql.NullFloat64 for empty float64 check
// func nullFloat64(value float64) sql.NullFloat64 {
// 	if value == 0 {
// 		return sql.NullFloat64{}
// 	}

//		return sql.NullFloat64{
//			Float64: value,
//			Valid:   true,
//		}
//	}
func generateMapFromStruct(instance interface{}, tag string) map[string]interface{} {
	result := make(map[string]interface{})

	val := reflect.Indirect(reflect.ValueOf(instance))
	typ := val.Type()

	for i := 0; i < val.NumField(); i++ {
		field := typ.Field(i)
		tag := field.Tag.Get(tag)
		if tag == "" {
			continue
		}

		fieldValue := val.Field(i).Interface()

		// Handle null types from "github.com/volatiletech/null/v9"
		switch v := fieldValue.(type) {
		case null.String:
			if v.Valid {
				result[tag] = v.String
			}
		case null.Int:
			if v.Valid {
				result[tag] = v.Int
			}
		case null.Int64:
			if v.Valid {
				result[tag] = v.Int64
			}
		case null.Uint8: // New support for null.Uint8
			if v.Valid {
				result[tag] = v.Uint8
			}
		case null.Int32:
			if v.Valid {
				result[tag] = v.Int32
			}
		case null.Uint32:
			if v.Valid {
				result[tag] = v.Uint32
			}
		case null.Uint64:
			if v.Valid {
				result[tag] = v.Uint64
			}
		case null.Float32:
			if v.Valid {
				result[tag] = v.Float32
			}
		case null.Float64:
			if v.Valid {
				result[tag] = v.Float64
			}
		case null.Bool:
			if v.Valid {
				result[tag] = v.Bool
			}
		case null.Time:
			if v.Valid {
				result[tag] = v.Time
			}
		default:
			// Handle other types normally
			result[tag] = fieldValue
		}
	}

	return result
}
