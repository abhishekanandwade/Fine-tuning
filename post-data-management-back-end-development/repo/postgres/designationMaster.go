package repo

import (
	"context"
	"time"

	"pmdm/core/domain"
	"pmdm/core/port"

	//"github.com/Masterminds/squirrel"
	sq "github.com/Masterminds/squirrel"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5"
	config "gitlab.cept.gov.in/it-2.0-common/api-config"
	dblib "gitlab.cept.gov.in/it-2.0-common/api-db"
)

// PostManagementRepository implements port.EstablishmentMasterRepository interface
// and provides access to the postgres database for establishment master-related operations
type DesignationMasterRepository struct {
	db  *dblib.DB
	cfg *config.Config
}

// NewPostManagementRepository creates a new EstablishmentMasterRepository instance
func NewDesignationMasterRepository(db *dblib.DB, cfg *config.Config) *DesignationMasterRepository {
	return &DesignationMasterRepository{
		db,
		cfg,
	}
}

// ListCadreMasterQuery retrieves cadre master details from the database based on the group code
func (de *DesignationMasterRepository) DesignationMasterByGroupAndCadreQuery(gctx *gin.Context, groupCode, cadreCode string, reqMetaData port.MetaDataRequest) ([]domain.DesignationMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), de.cfg.GetDuration("db.QueryTimeoutMed"))
	defer cancel()

	query := dblib.Psql.Select("designation", "group_name", "cadre_name", "cadre_id", "group_id", "designation_id").
		From("pmdm.designation_master").
		Where(sq.Eq{"group_id": groupCode, "cadre_id": cadreCode}).OrderBy("designation").
		Offset(uint64(reqMetaData.Skip * reqMetaData.Limit)).
		Limit(uint64(reqMetaData.Limit))

	return dblib.SelectRows(ctx, de.db, query, pgx.RowToStructByNameLax[domain.DesignationMaster])
}

// ListDesignationsQry retrieves the list of all entries in the designation_master table
func (dmr *DesignationMasterRepository) ListDesignationsQry(gctx *gin.Context) ([]domain.DesignationMaster, error) {
	// Create a context with a timeout to prevent hanging queries
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), dmr.cfg.GetDuration("db.QueryTimeoutMed"))
	defer cancel()

	// Define a query to select the fields 'designation_id' and 'designation_name' from the 'designation_master' table
	query := sq.Select("designation_id", "designation").
		From("pmdm.designation_master")

	// Convert the query to SQL and get the arguments
	return dblib.SelectRows(ctx, dmr.db, query, pgx.RowToStructByNameLax[domain.DesignationMaster])
}
func (de *DesignationMasterRepository) CadreListByGroupQuery(
	gctx *gin.Context, groupCode string, reqMetaData port.MetaDataRequest,
) ([]domain.CadreMasterNew, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), de.cfg.GetDuration("db.QueryTimeoutMed"))
	defer cancel()

	subQuery := dblib.Psql.Select("cadre_id", "cadre_name").
		From("pmdm.cadre_master").
		Where("cadre_id IN (SELECT unnest(string_to_array(cadre_id, ','))::int FROM pmdm.group_master WHERE group_id = $1)", groupCode).
		OrderBy("cadre_name").
		Offset(uint64(reqMetaData.Skip * reqMetaData.Limit)).
		Limit(uint64(reqMetaData.Limit))

	return dblib.SelectRows(ctx, de.db, subQuery, pgx.RowToStructByNameLax[domain.CadreMasterNew])
}
func (de *DesignationMasterRepository) DesignationListByCadreQuery(
	gctx *gin.Context, cadreCode string, reqMetaData port.MetaDataRequest,
) ([]domain.DesignationMasterNew, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), de.cfg.GetDuration("db.QueryTimeoutMed"))
	defer cancel()

	subQuery := dblib.Psql.Select("designation_id", "designation").
		From("pmdm.designation_master").
		Where(sq.Eq{"cadre_id": cadreCode}).
		OrderBy("designation_id").
		Offset(uint64(reqMetaData.Skip * reqMetaData.Limit)).
		Limit(uint64(reqMetaData.Limit))

	return dblib.SelectRows(ctx, de.db, subQuery, pgx.RowToStructByNameLax[domain.DesignationMasterNew])
}
func (dmr *DesignationMasterRepository) CreateDesignationMasterQuery(gctx *gin.Context, req domain.DesignationMaster) (*domain.DesignationMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), dmr.cfg.GetDuration("db.QueryTimeoutMed"))
	defer cancel()

	query := sq.Select(
		"designation_id",
	).
		From("pmdm.designation_master").
		Where(
			sq.Expr("designation_id = (SELECT MAX(designation_id) FROM pmdm.designation_master)"),
		)

	data, err := dblib.SelectOne(ctx, dmr.db, query, pgx.RowToStructByNameLax[domain.DesignationMaster])
	if err != nil {
		return nil, err
	}
	req.DesignationID = data.DesignationID + 1

	queryInsertTransaction := dblib.Psql.Insert("pmdm.designation_master").
		SetMap(generateMapFromStruct(req, "insert_designation_master")).
		Suffix("RETURNING designation_id,designation_uid")

	resp, err := dblib.InsertReturning(ctx, dmr.db, queryInsertTransaction, pgx.RowToStructByNameLax[domain.DesignationMaster])
	if err != nil {
		return nil, err
	}
	req.DesignationID = resp.DesignationID
	req.DesignationUID = resp.DesignationUID
	return &req, nil
}

// ListAllDesignationsQry retrieves the list of all entries in the designation_master table
func (dmr *DesignationMasterRepository) ListAllDesignationsQry(gctx *gin.Context, reqMetadata port.MetaDataRequest) ([]domain.DesignationMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), dmr.cfg.GetDuration("db.QueryTimeoutMed"))
	defer cancel()

	subQuery := sq.Select(
		"MAX(designation_uid) AS max_uid",
	).
		From("pmdm.designation_master").
		GroupBy("designation_id")

	query := sq.Select(
		"dm.designation_id",
		"dm.designation",
		"dm.group_name",
		"dm.cadre_name",
		"dm.valid_from",
		"dm.valid_to",
		"dm.status",
		"COALESCE(dm.remarks, '') as remarks",
		"dm.cadre_id",
		"dm.group_id",
		"dm.designation_uid",
	).
		FromSelect(subQuery, "sub").
		Join("pmdm.designation_master dm ON dm.designation_uid = sub.max_uid").
		OrderBy("dm.designation_uid").
		Offset(uint64(reqMetadata.Skip * reqMetadata.Limit)).
		Limit(uint64(reqMetadata.Limit))

	return dblib.SelectRows(ctx, dmr.db, query, pgx.RowToStructByNameLax[domain.DesignationMaster])
}

func (dmr *DesignationMasterRepository) UpdateDesignationMasterQuery(gctx *gin.Context, req domain.DesignationMaster) (*domain.DesignationMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), dmr.cfg.GetDuration("db.QueryTimeoutMed"))
	defer cancel()

	updateFields := generateMapFromStruct(req, "update_designation_master") // only non-empty fields

	queryUpdate := dblib.Psql.Update("pmdm.designation_master").
		SetMap(updateFields).
		Where(sq.Eq{"designation_uid": req.DesignationUID}).
		Suffix("RETURNING designation_uid")

	resp, err := dblib.UpdateReturning(ctx, dmr.db, queryUpdate, pgx.RowToStructByNameLax[domain.DesignationMaster])
	if err != nil {
		return nil, err
	}
	req.DesignationUID = resp.DesignationUID
	return &req, nil
}

func (dmr *DesignationMasterRepository) ListAllDesignationsD1(
	gctx *gin.Context,
	validFrom, validTo time.Time,
	reqMetadata port.MetaDataRequest,
) ([]domain.DesignationMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), dmr.cfg.GetDuration("db.QueryTimeoutMed"))
	defer cancel()

	query := dblib.Psql.
		Select(
			"COALESCE(dm.designation_id, 0) AS designation_id",
			"COALESCE(dm.designation, '') AS designation",
			"COALESCE(dm.group_name, '') AS group_name",
			"COALESCE(dm.cadre_name, '') AS cadre_name",
			"dm.valid_from",
			"dm.valid_to",
			"COALESCE(dm.status, '') AS status",
			"COALESCE(dm.remarks, '') AS remarks",
			"COALESCE(dm.cadre_id, 0) AS cadre_id",
			"COALESCE(dm.group_id, 0) AS group_id",
			"COALESCE(dm.designation_uid, 0) AS designation_uid",
		).
		From("pmdm.designation_master dm").
		Where(sq.And{
			sq.GtOrEq{"dm.created_date": validFrom},
			sq.LtOrEq{"dm.created_date": validTo},
		}).
		OrderBy("dm.designation_uid").
		Offset(uint64(reqMetadata.Skip * reqMetadata.Limit)).
		Limit(uint64(reqMetadata.Limit))

	return dblib.SelectRows(
		ctx,
		dmr.db,
		query,
		pgx.RowToStructByNameLax[domain.DesignationMaster],
	)
}
