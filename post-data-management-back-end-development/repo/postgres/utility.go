 package repo

// import (
// 	//"database/sql"
// 	//"fmt"

// 	"fmt"
// 	"reflect"

// 	//"sort"
// 	//"strconv"
// 	"strings"
// 	"time"

// 	sq "github.com/Masterminds/squirrel"
// 	"github.com/gin-gonic/gin"
// 	log "gitlab.cept.gov.in/it-2.0-common/api-log"

// 	"github.com/jackc/pgx/v5"
// 	"github.com/jackc/pgx/v5/pgconn"

// 	"context"
// 	//"reflect"
// 	//"github.com/Masterminds/squirrel"
// )

// type SQLValue string

// var errTooManyRows = fmt.Errorf("too many rows")

// func execReturn[T any](ctx context.Context, db *DB, sql string, args []any, scanFn pgx.RowToFunc[T], log *logger.Logger) (T, error) {
// 	var result T
// 	rows, err := db.Query(ctx, sql, args...)
// 	if err != nil {
// 		log.Error("pgutility, Err query at exec return:", err.Error())
// 		return result, err
// 	}
// 	defer rows.Close()
// 	collectedRow, err := pgx.CollectOneRow(rows, scanFn)
// 	if err != nil {
// 		log.Error("pgutility, Err collecting rows at exec run:", err.Error())
// 		return result, err
// 	}
// 	if rows.CommandTag().RowsAffected() > 1 {
// 		log.Error("pgutility,Too many Rows at exec run:", err)
// 		return collectedRow, errTooManyRows
// 	}

// 	return collectedRow, nil
// }

// func UpdateReturning[T any](ctx context.Context, db *DB, query sq.UpdateBuilder, scanFn pgx.RowToFunc[T]) (T, error) {
// 	var result T
// 	sql, args, err := query.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building sql at Update Returning:", err.Error())
// 		return result, err
// 	}
// 	collectedrows, err := execReturn(ctx, db, sql, args, scanFn)
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err collected rows at Update Returning:", err.Error())
// 		return result, err
// 	}
// 	return collectedrows, nil

// }

// func execinsert(ctx context.Context, db *DB, sql string, args []any) (pgconn.CommandTag, error) {
// 	//log.Debug("came inside exec")
// 	rows, err := db.Exec(ctx, sql, args...)

// 	if err != nil {
// 		log.Error(nil,"pgutility, Err running query at exec:", err.Error())
// 		return rows, err
// 	}
// 	//defer rows.Close()
// 	return rows, err
// }

// func execupdate(ctx context.Context, db *DB, sql string, args []any) (pgconn.CommandTag, error) {
// 	//log.Debug("came inside exec")
// 	rows, err := db.Exec(ctx, sql, args...)

// 	if err != nil {
// 		log.Error(nil,"pgutility, Err running query at exec:", err.Error())
// 		return rows, err
// 	}
// 	//defer rows.Close()
// 	return rows, err
// }

// func execdelete(ctx context.Context, db *DB, sql string, args []any) (pgconn.CommandTag, error) {
// 	//log.Debug("came inside exec")
// 	rows, err := db.Exec(ctx, sql, args...)

// 	if err != nil {
// 		log.Error(nil,"pgutility, Err running query at exec:", err.Error())
// 		return rows, err
// 	}
// 	//defer rows.Close()
// 	return rows, err
// }

// func exec(ctx context.Context, db *DB, sql string, args []any) (pgconn.CommandTag, error) {
// 	log.Debug(nil,"came inside exec")
// 	rows, err := db.Query(ctx, sql, args...)
// 	log.Debug(nil,"Rows:", rows.RawValues())
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err running query at exec:", err.Error())
// 		return pgconn.CommandTag{}, err
// 	}
// 	defer rows.Close()
// 	return rows.CommandTag(), rows.Err()
// }

// func Update(ctx context.Context, db *DB, query sq.UpdateBuilder) (pgconn.CommandTag, error) {
// 	sql, args, err := query.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building sql at Update:", err.Error())
// 		return pgconn.CommandTag{}, err
// 	}

// 	//log.Debug("sql:", sql, "args:", args)

// 	return execupdate(ctx, db, sql, args)
// }

// func Delete(ctx context.Context, db *DB, query sq.DeleteBuilder) (pgconn.CommandTag, error) {
// 	sql, args, err := query.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building sql at Delete:", err.Error())
// 		return pgconn.CommandTag{}, err
// 	}
// 	return execdelete(ctx, db, sql, args)
// }

// func Insert(ctx context.Context, db *DB, query sq.InsertBuilder) (pgconn.CommandTag, error) {
// 	sql, args, err := query.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building sql at Insert:", err.Error())
// 		return pgconn.CommandTag{}, err
// 	}

// 	//log.Debug("sql:", sql, "args:", args)
// 	return execinsert(ctx, db, sql, args)
// }

// func ExecRow(ctx context.Context, db *DB,sql string, args ...any) (pgconn.CommandTag, error) {
// 	ct, err := exec(ctx, db, sql, args)
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err exec at ExecRow:", err.Error())
// 		return ct, err
// 	}
// 	rowsAffected := ct.RowsAffected()
// 	if rowsAffected == 0 {
// 		return ct, pgx.ErrNoRows
// 	} else if rowsAffected > 1 {
// 		log.Error(nil,"pgutility, Too Many Rows at ExecRow:", err)
// 		return ct, errTooManyRows
// 	}
// 	return ct, nil
// }

// func SelectOneOK[T any](ctx context.Context, db *DB, builder sq.SelectBuilder, scanFn pgx.RowToFunc[T]) (T, bool, error) {

// 	var zero T
// 	sql, args, err := builder.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building sql at SelectOne:", err.Error())
// 		return zero, false, err
// 	}
// 	rows, err := db.Query(ctx, sql, args...)
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err running query at SelectOne:", err.Error())
// 		return zero, false, err
// 	}
// 	defer rows.Close()
// 	collectedRow, b, err := CollectOneRowOK(rows, scanFn)
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err collecting rows at SelectOne:", err.Error())
// 		return zero, false, err
// 	}

// 	if rows.CommandTag().RowsAffected() > 1 {
// 		log.Error(nil,"pgutility, Too Many Rows at SelectOne:", err)
// 		return collectedRow, false, errTooManyRows
// 	}
// 	return collectedRow, b, nil
// }

// func SelectOne[T any](ctx context.Context, db *DB, builder sq.SelectBuilder, scanFn pgx.RowToFunc[T]) (T, error) {
// 	var zero T
// 	sql, args, err := builder.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building sql at SelectOne:", err.Error())
// 		return zero, err
// 	}
// 	rows, err := db.Query(ctx, sql, args...)
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err running query at SelectOne:", err.Error())
// 		return zero, err
// 	}
// 	defer rows.Close()

// 	collectedRow, err := pgx.CollectOneRow(rows, scanFn)
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err collecting rows at SelectOne:", err.Error())
// 		return zero, err
// 	}

// 	if rows.CommandTag().RowsAffected() > 1 {
// 		log.Error(nil,"pgutility, Too Many Rows at SelectOne:", err)
// 		return collectedRow, errTooManyRows
// 	}
// 	return collectedRow, nil
// }

// func InsertReturning[T any](ctx context.Context, db *DB, builder sq.InsertBuilder, scanFn pgx.RowToFunc[T]) (T, error) {
// 	var zero T
// 	sql, args, err := builder.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building utility at InsertReturning:", err.Error())
// 		return zero, err
// 	}
// 	collectedRow, err := execReturn(ctx, db, sql, args, scanFn)
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err Collected Row at InsertReturning:", err.Error())
// 		return zero, err
// 	}
// 	return collectedRow, nil

// }

// func SelectRows1[T any](ctx context.Context, db *DB, builder sq.SelectBuilder, scanFn pgx.RowToFunc[T]) ([]T, error) {
// 	sql, args, err := builder.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building sql at Select Rows :", err.Error())
// 		return nil, err
// 	}
// 	rows, _ := db.Query(ctx, sql, args...)
// 	log.Debug(nil,"Descriptions", rows.FieldDescriptions())
// 	log.Debug(nil,"Values", rows.RawValues)
// 	log.Debug(nil,"command Tag", rows.CommandTag())

// 	//pgx.CollectOneRow(rows,pgx.RowToMap)
// 	//return pgx.CollectRows(rows, pgx.RowTo[domain.Bag])

// 	collectedRows, err := pgx.CollectRows(rows, scanFn)
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err collecting Rows at Select Rows :", err.Error())
// 		return nil, err
// 	}

// 	return collectedRows, nil
// }
// func SelectRows[T any](ctx context.Context, db *DB, builder sq.SelectBuilder, scanFn pgx.RowToFunc[T]) ([]T, error) {
// 	sql, args, err := builder.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building sql at Select Rows :", err.Error())
// 		return nil, err
// 	}
// 	rows, _ := db.Query(ctx, sql, args...)

// 	defer rows.Close()
// 	collectedRows, err := pgx.CollectRows(rows, scanFn)
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err collecting Rows at Select Rows :", err.Error())
// 		return nil, err
// 	}

// 	return collectedRows, nil
// }

// func SelectRowsTag[T any](ctx context.Context, db *DB, builder sq.SelectBuilder, scanFn pgx.RowToFunc[T],tag string) ([]T, error) {
// 	structTagKey = tag
// 	sql, args, err := builder.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building sql at Select Rows :", err.Error())
// 		return nil, err
// 	}
// 	rows, _ := db.Query(ctx, sql, args...)
// 	defer rows.Close()
// 	collectedRows, err := pgx.CollectRows(rows, scanFn)
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err collecting Rows at Select Rows :", err.Error())
// 		return nil, err
// 	}

// 	return collectedRows, nil
// }

// var structTagKey string

// func RowToStructByTag[T any](row pgx.CollectableRow) (T, error) {
// 	//structTagKey=tagkey
// 	var value T
// 	err := row.Scan(&tagStructRowScanner{ptrToStruct: &value, lax: true})
// 	return value, err
// }

// type tagStructRowScanner struct {
// 	ptrToStruct any
// 	lax         bool
// }

// func (ts *tagStructRowScanner) ScanRow(rows pgx.Rows) error {

// 	dst := ts.ptrToStruct
// 	dstValue := reflect.ValueOf(dst)
// 	if dstValue.Kind() != reflect.Ptr {
// 		return fmt.Errorf("dst not a pointer")
// 	}

// 	dstElemValue := dstValue.Elem()

// 	scanTargets, err := ts.appendScanTargets(dstElemValue, nil, rows.FieldDescriptions(), structTagKey)

// 	if err != nil {

// 		return err
// 	}

// 	for i, t := range scanTargets {

// 		if t == nil {
// 			return fmt.Errorf("struct doesn't have corresponding field to match returned column %s", rows.FieldDescriptions()[i].Name)
// 		}
// 	}

// 	return rows.Scan(scanTargets...)
// }

// func (rs *tagStructRowScanner) appendScanTargets(dstElemValue reflect.Value, scanTargets []any, fldDescs []pgconn.FieldDescription, tagkey string) ([]any, error) {
// 	var err error
// 	dstElemType := dstElemValue.Type()

// 	if scanTargets == nil {
// 		scanTargets = make([]any, len(fldDescs))
// 	}

// 	for i := 0; i < dstElemType.NumField(); i++ {
// 		sf := dstElemType.Field(i)

// 		if sf.PkgPath != "" && !sf.Anonymous {

// 			// Field is unexported, skip it.
// 			continue
// 		}

// 		// Handle anoymous struct embedding, but do not try to handle embedded pointers.
// 		if sf.Anonymous && sf.Type.Kind() == reflect.Struct {

// 			scanTargets, err = rs.appendScanTargets(dstElemValue.Field(i), scanTargets, fldDescs, tagkey)
// 			if err != nil {
// 				return nil, err
// 			}
// 		} else {

// 			dbTag, dbTagPresent := sf.Tag.Lookup(structTagKey)
// 			if dbTagPresent {

// 				dbTag = strings.Split(dbTag, ",")[0]
// 			}
// 			if dbTag == "-" {

// 				// Field is ignored, skip it.
// 				continue
// 			}

// 			colName := dbTag

// 			if !dbTagPresent {

// 				colName = sf.Name
// 			}

// 			fpos := fieldPosByName(fldDescs, colName)

// 			if fpos == -1 {
// 				if rs.lax {

// 					continue
// 				}
// 				return nil, fmt.Errorf("cannot find field %s in returned row", colName)
// 			}
// 			if fpos >= len(scanTargets) && !rs.lax {
// 				return nil, fmt.Errorf("cannot find field %s in returned row", colName)
// 			}

// 			scanTargets[fpos] = dstElemValue.Field(i).Addr().Interface()
// 		}
// 	}

// 	return scanTargets, err
// }

// func fieldPosByName(fldDescs []pgconn.FieldDescription, field string) (i int) {
// 	i = -1
// 	for i, desc := range fldDescs {
// 		if strings.EqualFold(desc.Name, field) {
// 			return i
// 		}
// 	}
// 	return
// }

// // func withTx( f func(ctx context.Context, tx pgx.Tx)) {
// // 	ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
// // 	defer cancel()

// // 	conn := connectPG(t, ctx)
// // 	defer closeConn(t, conn)

// // 	tx, err := conn.Begin(ctx)
// // 	require.NoError(t, err)
// // 	defer tx.Rollback(ctx)

// // 	f(ctx, tx)
// // }

// //pgxutil.QueueUpdateRowReturning(batch, pgx.Identifier{"t"}, map[string]any{"age": 70}, map[string]any{"name": "John"}, "*", pgx.RowToAddrOfStructByPos[Person], &person)

// func StructToSetMap(article interface{}) map[string]interface{} {

// 	setMap := make(map[string]interface{})
// 	//type Val struct{}
// 	// Using reflection to iterate over the struct fields
// 	//reflect.Indirect()
// 	val := reflect.ValueOf(article).Elem()

// 	for i := 0; i < val.NumField(); i++ {
// 		field := val.Type().Field(i)
// 		tag := field.Tag.Get("json")

// 		// Skip fields without the "db" tag
// 		if tag == "" {
// 			continue
// 		}

// 		// Check if the value is the zero value for its type
// 		switch val.Field(i).Kind() {
// 		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
// 			if val.Field(i).Int() == 0 {
// 				continue
// 			}
// 		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
// 			if val.Field(i).Uint() == 0 {
// 				continue
// 			}
// 		case reflect.Float32, reflect.Float64:
// 			if val.Field(i).Float() == 0 {
// 				continue
// 			}
// 		case reflect.String:
// 			if val.Field(i).String() == "" {
// 				continue
// 			}
// 		case reflect.Bool:
// 			if !val.Field(i).Bool() {
// 				continue
// 			}

// 		case reflect.Struct:
// 			if val.Field(i).Type() == reflect.TypeOf(time.Time{}) && val.Field(i).Interface().(time.Time).IsZero() {
// 				continue
// 			}

// 		default:
// 			// Handle other types as needed
// 		}

// 		setMap[tag] = val.Field(i).Interface()
// 	}

// 	return setMap
// }
// func QueueExecRow(batch *pgx.Batch, builder sq.Sqlizer) error {
// 	var qErr error

// 	sql, args, err := builder.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building sql at batch return :", err.Error())
// 		return err
// 	}
// 	batch.Queue(sql, args...).Exec(func(ct pgconn.CommandTag) error {
// 		rowsAffected := ct.RowsAffected()
// 		if rowsAffected == 0 {
// 			qErr = pgx.ErrNoRows
// 			return nil
// 		}
// 		// } else if rowsAffected > 1 {
// 		// 	return errTooManyRows
// 		// }
// 		return nil
// 	})

// 	return qErr
// }

// func QueueReturn[T any](batch *pgx.Batch, builder sq.Sqlizer, scanFn pgx.RowToFunc[T],result *[]T) error {

// 	var qErr error

// 	sql, args, err := builder.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building sql at batch return :", err.Error())
// 		return err
// 	}
// 	batch.Queue(sql, args...).Query(func(rows pgx.Rows) error {
// 		collectedRows, err := pgx.CollectRows(rows, scanFn)
// 		if err != nil {
// 			qErr = err
// 			return nil
// 		}
// 		*result = collectedRows
// 		return nil
// 	})

// 	return qErr
// }

// func QueueReturnRow[T any](batch *pgx.Batch, builder sq.Sqlizer, scanFn pgx.RowToFunc[T],result *T) error {
// 	var qErr error
// 	//log.Debug("Came inside")

// 	sql, args, err := builder.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building sql at queue return Row :", err.Error())
// 		return err
// 	}

// 	batch.Queue(sql, args...).Query(func(rows pgx.Rows) error {
// 		collectedRow, err := pgx.CollectOneRow(rows, scanFn)
// 		if err != nil {
// 			log.Debug(nil,"Error collect one row:", err.Error())
// 			qErr = err
// 			return nil
// 		}

// 		// if rows.CommandTag().RowsAffected() > 1 {
// 		// 	//log.Debug("Came to rows affected:", err)
// 		// 	qErr = errTooManyRows
// 		// 	return nil
// 		// }

// 		*result = collectedRow
// 		return nil
// 	})
// 	//log.Debug("qerr is:", qErr)
// 	return qErr
// }

// func TxReturnRow[T any](ctx context.Context, tx pgx.Tx, builder sq.Sqlizer, scanFn pgx.RowToFunc[T],result *T) error {
// 	//var zero T
// 	sql, args, err := builder.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building sql at TxReturnRow:", err.Error())
// 		return err
// 	}
// 	rows, err := tx.Query(ctx, sql, args...)
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err running query at TxReturnRow:", err.Error())
// 		return err
// 	}
// 	defer rows.Close()

// 	collectedRow, err := pgx.CollectOneRow(rows, scanFn)
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err collecting rows at TxReturnRow:", err.Error())
// 		return err
// 	}
// 	*result = collectedRow
// 	return nil
// }

// func TxRows[T any](ctx context.Context, tx pgx.Tx, builder sq.Sqlizer, scanFn pgx.RowToFunc[T],result *[]T) error {
// 	//var zero T
// 	sql, args, err := builder.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building sql at TxReturnRow:", err.Error())
// 		return err
// 	}
// 	rows, err := tx.Query(ctx, sql, args...)
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err running query at TxRows:", err.Error())
// 		return err
// 	}
// 	defer rows.Close()

// 	collectedRows, err := pgx.CollectRows(rows, scanFn)
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err collecting rows at TxRows:", err.Error())
// 		return err
// 	}

// 	*result = collectedRows
// 	return nil
// }

// func TxExec(ctx context.Context, tx pgx.Tx, builder sq.Sqlizer) error {
// 	sql, args, err := builder.ToSql()
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err building sql at Txexec:", err.Error())
// 		return err
// 	}
// 	_, err = tx.Exec(ctx, sql, args...)
// 	if err != nil {
// 		log.Error(nil,"pgutility, Err running query at Txexec:", err.Error())
// 		return err
// 	}
// 	return nil
// }

// func generateMapFromStruct(instance interface{}, tag string) map[string]interface{} {
// 	result := make(map[string]interface{})

// 	val := reflect.Indirect(reflect.ValueOf(instance))
// 	typ := val.Type()

// 	for i := 0; i < val.NumField(); i++ {
// 		field := typ.Field(i)
// 		tag := field.Tag.Get(tag)
// 		if tag != "" {
// 			result[tag] = val.Field(i).Interface()
// 		}
// 	}
// 	return result
// }

// func generateColumnsFromStruct(instance interface{}, tag string) []string {
// 	var columns []string

// 	val := reflect.Indirect(reflect.ValueOf(instance))
// 	typ := val.Type()

// 	for i := 0; i < val.NumField(); i++ {
// 		field := typ.Field(i)
// 		tag := field.Tag.Get(tag)
// 		if tag != "" {
// 			columns = append(columns, tag)
// 		}
// 	}

// 	return columns
// }

// func CollectOneRowOK[T any](rows pgx.Rows, fn pgx.RowToFunc[T]) (T, bool, error) {
// 	var value T
// 	var err error
// 	value, err = pgx.CollectOneRow(rows, fn)
// 	if err != nil {
// 		if err == pgx.ErrNoRows {
// 			return value, false, nil
// 		}
// 		return value, false, err
// 	}
// 	return value, true, nil
// }

// func Tx(gctx *gin.Context, dbPool *DB, f func(ctx context.Context, gctx *gin.Context, tx pgx.Tx, params ...interface{}) error, params ...interface{}) error {
// 	//func withTx1(ctx context.Context, dbPool *pgxpool.Pool, f func(ctx context.Context, tx pgx.Tx, params ...interface{}) params ...interface{},error) error {
// 	//var cancel context.CancelFunc
// 	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
// 	defer cancel()

// 	tx, err := dbPool.Begin(ctx)
// 	if err != nil {
// 		return fmt.Errorf("error beginning transaction: %w", err)
// 	}
// 	defer tx.Rollback(ctx) // Rollback if not committed

// 	if err := f(ctx, gctx, tx, params...); err != nil {
// 		// If an error occurred during the transactional logic, rollback
// 		return fmt.Errorf("error in transactional logic: %w", err)
// 	}

// 	// Commit the transaction
// 	if err := tx.Commit(ctx); err != nil {
// 		return fmt.Errorf("error committing transaction: %w", err)
// 	}

// 	return nil
// }
