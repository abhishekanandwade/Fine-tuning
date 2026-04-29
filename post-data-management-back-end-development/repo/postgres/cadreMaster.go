package repo

import (
	"context"
	"strconv"
	"time"

	// "pmdm/api-config"
	"pmdm/core/domain"
	"pmdm/core/port"

	//"github.com/Masterminds/squirrel"
	sq "github.com/Masterminds/squirrel"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5"
	config "gitlab.cept.gov.in/it-2.0-common/api-config"
	dblib "gitlab.cept.gov.in/it-2.0-common/api-db"
)

// CadreMasterRepository implements port.CadreMasterRepository interface
// and provides access to the postgres database for cadre master-related operations
type CadreMasterRepository struct {
	db  *dblib.DB
	cfg *config.Config
}

// NewCadreMasterRepository creates a new CadreMasterRepository instance
func NewCadreMasterRepository(db *dblib.DB, cfg *config.Config) *CadreMasterRepository {
	return &CadreMasterRepository{
		db,
		cfg,
	}
}

// ListCadreMasterQuery retrieves cadre master details from the database based on the group code
func (cmr *CadreMasterRepository) CadreMasterByGroupCodeQuery(gctx *gin.Context, groupCode string, reqMetaData port.MetaDataRequest) ([]domain.CadreMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), cmr.cfg.GetDuration("db.QueryTimeoutLow"))
	defer cancel()

	query := dblib.Psql.Select(
		"cadre_id",
		"TRIM(cadre_name) AS cadre_name",
		"TRIM(group_name) AS group_name").
		From("pmdm.cadre_master").
		Where(sq.Eq{"group_id": groupCode}).
		OrderBy("cadre_id").
		Offset(uint64(reqMetaData.Skip * reqMetaData.Limit)).
		Limit(uint64(reqMetaData.Limit))

	return dblib.SelectRows(ctx, cmr.db, query, pgx.RowToStructByNameLax[domain.CadreMaster])
}

func (cmr *CadreMasterRepository) CreateCadreMasterQuery(gctx *gin.Context, req domain.CadreMaster) (*domain.CadreMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), cmr.cfg.GetDuration("db.QueryTimeoutMed"))
	defer cancel()

	queryInsertTransaction := dblib.Psql.Insert("pmdm.cadre_master").
		SetMap(generateMapFromStruct(req, "insert_cadre_master")).
		Suffix("RETURNING cadre_id")

	resp, err := dblib.InsertReturning(ctx, cmr.db, queryInsertTransaction, pgx.RowToStructByNameLax[domain.CadreMaster])
	if err != nil {
		return nil, err
	}
	req.CadreID.Int32 = resp.CadreID.Int32
	cadreIDStr := strconv.FormatInt(int64(req.CadreID.Int32), 10)

	pattern := "%," + cadreIDStr + ",%"
	queryUpdateGroupMaster := dblib.Psql.Update("pmdm.group_master").
		Set("cadre_id", sq.Case().
			When(
				sq.Expr("NOT (',' || cadre_id || ',' LIKE ?)", pattern),
				sq.Expr("CONCAT(cadre_id, ',' || ?)", cadreIDStr),
			).
			Else(sq.Expr("cadre_id::text")),
		).
		Where(sq.Eq{"group_id": req.GroupCode.Int16})

	_, err = dblib.Update(ctx, cmr.db, queryUpdateGroupMaster)
	if err != nil {
		return nil, err
	}

	return &req, nil
}

// ListCadresQry retrieves the list of all entries in the cadre_master table
func (cmr *CadreMasterRepository) ListCadresQry(gctx *gin.Context, reqMetadata port.MetaDataRequest) ([]domain.CadreMaster, error) {
	// Create a context with a timeout to prevent hanging queries
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), cmr.cfg.GetDuration("db.QueryTimeoutMed"))
	defer cancel()

	// Define a query to select the fields 'cadre_id' and 'cadre_name' from the 'cadre_master' table
	query := sq.Select("cadre_id", "cadre_name", "group_name").
		From("pmdm.cadre_master").
		OrderBy("cadre_id").
		Offset(uint64(reqMetadata.Skip * reqMetadata.Limit)).
		Limit(uint64(reqMetadata.Limit))

	return dblib.SelectRows(ctx, cmr.db, query, pgx.RowToStructByNameLax[domain.CadreMaster])
}

// ListAllCadresQry retrieves the list of all entries in the cadre_master table
func (cmr *CadreMasterRepository) ListAllCadresQry(gctx *gin.Context, reqMetadata port.MetaDataRequest) ([]domain.CadreMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), cmr.cfg.GetDuration("db.QueryTimeoutMed"))
	defer cancel()

	query := sq.Select(
		"cadre_id",
		"cadre_name",
		"group_name",
		"pay_level",
		"grade_pay",
		"valid_from",
		"valid_to",
		"status",
		"remarks",
		"group_id AS group_code").
		From("pmdm.cadre_master").
		OrderBy("cadre_id").
		Offset(uint64(reqMetadata.Skip * reqMetadata.Limit)).
		Limit(uint64(reqMetadata.Limit))

	return dblib.SelectRows(ctx, cmr.db, query, pgx.RowToStructByNameLax[domain.CadreMaster])
}

func (cmr *CadreMasterRepository) UpdateCadreMasterQuery(gctx *gin.Context, req domain.CadreMaster) (*domain.CadreMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), cmr.cfg.GetDuration("db.QueryTimeoutMed"))
	defer cancel()

	updateFields := generateMapFromStruct(req, "update_cadre_master") // only non-empty fields

	queryUpdate := dblib.Psql.Update("pmdm.cadre_master").
		SetMap(updateFields).
		Where(sq.Eq{"cadre_id": req.CadreID}).
		Suffix("RETURNING cadre_id")

	resp, err := dblib.UpdateReturning(ctx, cmr.db, queryUpdate, pgx.RowToStructByNameLax[domain.CadreMaster])
	if err != nil {
		return nil, err
	}
	req.CadreID = resp.CadreID
	return &req, nil
}

func (cmr *CadreMasterRepository) ListAllCadresQryD1(
	gctx *gin.Context,
	validFrom, validTo time.Time,
	reqMetadata port.MetaDataRequest,
) ([]domain.CadreMasterD1, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), cmr.cfg.GetDuration("db.QueryTimeoutMed"))
	defer cancel()

	query := sq.
		Select(
			"COALESCE(cadre_id, 0) AS cadre_id",
			"COALESCE(cadre_name, '') AS cadre_name",
			"COALESCE(group_name, '') AS group_name",
			"COALESCE(pay_level, 0) AS pay_level",
			"COALESCE(grade_pay, 0) AS grade_pay",
			"valid_from",
			"valid_to",
			"COALESCE(status, '') AS status",
			"COALESCE(remarks, '') AS remarks",
			"COALESCE(group_id, 0) AS group_id",
		).
		From("pmdm.cadre_master").
		Where(sq.And{
			sq.GtOrEq{"updated_date": validFrom},
			sq.LtOrEq{"updated_date": validTo},
		}).
		OrderBy("cadre_id").
		Offset(uint64(reqMetadata.Skip * reqMetadata.Limit)).
		Limit(uint64(reqMetadata.Limit)).
		PlaceholderFormat(sq.Dollar)

	return dblib.SelectRows(
		ctx,
		cmr.db,
		query,
		pgx.RowToStructByNameLax[domain.CadreMasterD1],
	)
}
