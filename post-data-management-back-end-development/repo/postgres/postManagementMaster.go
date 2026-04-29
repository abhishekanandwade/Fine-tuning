package repo

import (
	"context"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"strings"

	"pmdm/core/domain"
	"pmdm/core/port"
	"strconv"
	"time"

	//"github.com/Masterminds/squirrel"

	sq "github.com/Masterminds/squirrel"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5"

	"github.com/minio/minio-go/v7"
	config "gitlab.cept.gov.in/it-2.0-common/api-config"
	dblib "gitlab.cept.gov.in/it-2.0-common/api-db"

	//apierrors "gitlab.cept.gov.in/it-2.0-common/api-errors"
	"github.com/volatiletech/null/v9"
	log "gitlab.cept.gov.in/it-2.0-common/api-log"
)

// PostManagementRepository implements port.EstablishmentMasterRepository interface
// and provides access to the postgres database for establishment master-related operations
type PostManagementRepository struct {
	db          *dblib.DB
	cfg         *config.Config
	MinioClient *minio.Client
}

// NewPostManagementRepository creates a new EstablishmentMasterRepository instance
func NewPostManagementRepository(db *dblib.DB, cfg *config.Config, MinioClient *minio.Client) *PostManagementRepository {
	return &PostManagementRepository{
		db,
		cfg,
		MinioClient,
	}
}

func generateUniqueID() string {
	return strconv.FormatInt(time.Now().UnixNano(), 10)
}

func (pmr *PostManagementRepository) PostManagementByOfficeIDQuery(gctx *gin.Context, officeID int) ([]domain.PostManagementMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()
	//log.Error(ctx,ContextWithTimeout)

	query := dblib.Psql.Select(
		PMOfficeID, // assuming office_id is an integer, use 0 as default
		PMPostName,
		PMOfficeName,
		PMGroupID, // assuming group_id is an integer, use 0 as default
		PMFilledStatus,
		PMPostID, // assuming post_id is an integer, use 0 as default
		PMDestinaton,
		PMPermanentStatus,
		PMStatus,
		PMGroupName,
		PMEmployeeGroup,
		PMCradeID, // assuming cadre_id is an integer, use 0 as default
		PMCradeName,
		"COALESCE(pay_level, 0) AS pay_level", // assuming pay_level is an integer, use 0 as default
		"COALESCE(grade_pay, 0) AS grade_pay", // assuming grade_pay is an integer, use 0 as default
		"COALESCE(designation_id, 0) AS designation_id", // assuming designation_id is an integer, use 0 as default
	).
		From(PMPostManagementMaster).
		//Where(sq.Eq{"office_id": officeID})
		Where(sq.And{sq.Eq{"office_id": officeID}, sq.Eq{"status": "Active"}}) // Adding status condition

	//log.Println(QueryConstruted)

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaster])
}

// PostManagementByOfficeIDQuery retrieves post management master details from the database based on the office ID
func (pmr *PostManagementRepository) PostManagementByOfficeIDQueryMDW(gctx *gin.Context, officeID int, raPostID int64, reqMetadata port.MetaDataRequest) ([]domain.PostManagementMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select(QOfficeID, QPostName, QOfficeName, QGroupID, QFilledStatus, QPostID, QDestination, QPermanentStatus, QStatus).
		From(PMPostManagementMaster).
		Join("pmdm.post_mapping_detail pmd on pmd.employee_post_id=pm.post_id and pmd.role_authority=" + strconv.Itoa(int(raPostID))).
		Where(sq.Eq{"employee_office_id": officeID}).
		Where(sq.Eq{"status": "Active"}).
		OrderBy("office_id").
		Offset(uint64(reqMetadata.Skip * reqMetadata.Limit)).
		Limit(uint64(reqMetadata.Limit))
	//Where(sq.Eq{"office_id": officeID})
	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaster])
}

// PostManagementByOfficeIDAndStatusQuery retrieves post management master details from the database based on the office ID and filled status
func (pmr *PostManagementRepository) PostManagementByOfficeIDAndStatusQuery(gctx *gin.Context, officeID int, filledStatus string, groupID int) ([]domain.PostManagementMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select(
		"office_id",
		"post_id",
		"post_name",
		"cadre_name",
		"designation",
		"filled_status",
		"status",
		"group_id",
		"COALESCE(employee_group, '') AS employee_group",
		"COALESCE(cadre_id, 0) AS cadre_id").
		From(PostManagementMaster).
		Where(sq.Eq{"office_id": officeID}).
		Where(sq.Eq{"group_id": groupID})

	if !(filledStatus == "all" || filledStatus == "ALL") {
		query = query.Where(sq.Eq{"filled_status": filledStatus})
	}

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaster])
}

// PostManagementByCadreAndOfficeQuery retrieves post management master details from the database based on the Cadre Name
func (pmr *PostManagementRepository) PostManagementByCadreAndOfficeQuery(gctx *gin.Context, cadreName string, officeID int, reqMetadata port.MetaDataRequest) ([]domain.PostManagementMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select("office_id", "post_id", "post_name", "cadre_name", "post_status", "filled_status", "status").
		From(PostManagementMaster).
		Where(sq.Eq{"cadre_name": cadreName, "office_id": officeID}).
		OrderBy("office_id").
		Offset(uint64(reqMetadata.Skip * reqMetadata.Limit)).
		Limit(uint64(reqMetadata.Limit))

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaster])
}
func (pmr *PostManagementRepository) PostManagementGroupByCadreCountByOfficeID(gctx *gin.Context, officeID int) ([]domain.PostManagementMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select(("COUNT(*) AS count"), "cadre_name", "filled_status").
		From(PostManagementMaster).
		Where(sq.Eq{"office_id": officeID}).
		GroupBy("cadre_name", "filled_status")
	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaster])
}

// EstblishnentRegisterByOfficeQuery retrieves post management master details from the database based on the office ID
func (pmr *PostManagementRepository) EstblishnentRegisterByOfficeQuery(gctx *gin.Context, officeID int) (domain.PostManagementMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	//log.Println("Creating SQL select query for establishment register by office")

	query := dblib.Psql.Select(
		"COALESCE(office_id, 0) AS office_id",
		"COALESCE(office_name, '') AS office_name",
		"COALESCE(establishment_register_id, 0) AS establishment_register_id",
		"COALESCE(establishment_register_name, '') AS establishment_register_name",
		"COALESCE(sanctioned_strength, 0) AS sanctioned_strength",
	).
		From("pmdm.establishment_master").
		Where(sq.Eq{"office_id": officeID}).
		Limit(1)

	post, err := dblib.SelectOne(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaster])
	if err != nil {
		return domain.PostManagementMaster{}, err
	}
	return post, nil
}

func (pmr *PostManagementRepository) CreateEstablishmentRegister(gctx *gin.Context, establishment domain.PostManagementMaster) (*domain.PostManagementMaster1, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Insert("pmdm.establishment_master").
		Columns("office_id", "office_name", "establishment_register_name", "created_by", "created_date", "status").
		Values(establishment.OfficeID, establishment.OfficeName, establishment.EstablishmentRegisterName, establishment.CreatedBy, establishment.CreatedOn, establishment.Status).
		Suffix("RETURNING establishment_register_id, office_id, office_name, establishment_register_name, created_by, created_date, status")

	return dblib.InsertReturning(ctx, pmr.db, query, pgx.RowToAddrOfStructByPos[domain.PostManagementMaster1])
}
func (pmr *PostManagementRepository) isApproveStatusPending(ctx context.Context, postID int) (bool, error) {

	var pending bool
	err := pmr.db.QueryRow(ctx, "SELECT EXISTS(SELECT 1 FROM pmdm.post_management_master_maker WHERE post_id = $1 AND approve_status = 'Pending')", postID).Scan(&pending)
	if err != nil {
		return false, err
	}
	return pending, nil
}
func (pmr *PostManagementRepository) CreatePostManagementMakerQuery(gctx *gin.Context, posts []domain.PostManagementMaker) ([]*domain.PostManagementMaster3, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	var filled_status = "Vacant"
	var post_status = "Active"
	var valid_to = "9999-12-31T15:45:00Z"
	var masterMakerID string

	var response []*domain.PostManagementMaster3
	var post *domain.PostManagementMaster3

	log.Debug(ctx, "Starting to construct the SQL query for inserting post management maker records")

	query := dblib.Psql.Insert("pmdm.post_management_master_maker").
		Columns(
			"office_id",
			"post_name",
			"office_name",
			"group_id",
			"cadre_id",
			"cadre_name",
			"filled_status",
			"post_status",
			"allowances_attached",
			"allowance_description",
			"created_by",
			"created_date",
			"status",
			"valid_from",
			"valid_to",
			"order_casemark",
			"order_date",
			"upload_order_doc_name",
			"establishment_register_id",
			"designation",
			"pay_level",
			"grade_pay",
			"permanent_status",
			"establishment_register_name",
			"employee_group",
			"sanctioned_strength",
			"post_id",
			"approve_status",
			"approve_post_id",
			"new_office_id",
			"new_office_name",
			"designation_id",
			"exchange_post_id",
			"remarks",
			"master_maker_id",
		)
	for _, post := range posts {
		// Check if approve_status is pending for the given post_id
		pending, err := pmr.isApproveStatusPending(ctx, post.PostID)
		if err != nil {
			log.Error(ctx, "Error checking approve status", "error", err)
			return nil, err
		}
		if pending {
			// Throw an error indicating modification is already pending
			return nil, errors.New("modification is already pending for the given post_id")
		}
		// Ensure approve_status is only "Pending"
		if post.ApproveStatus != "Pending" {
			log.Error(ctx, "Invalid approve status, only 'Pending' is accepted", "post_id", post.PostID, "approve_status", post.ApproveStatus)
			return nil, errors.New("invalid approve status, only 'Pending' is accepted")
		}
		// Log post information before inserting
		log.Debug(ctx, "Preparing to insert post management maker record", "post", post)

		queryGetMasterMakerID := `
		SELECT COALESCE(master_maker_id, '') AS master_maker_id
	    FROM pmdm.post_management_master
	    WHERE post_id = $1`
		err = pmr.db.QueryRow(ctx, queryGetMasterMakerID, post.PostID).Scan(&masterMakerID)
		if err != nil {
			log.Error(gctx, "Failed to retrieve master_maker_id", "postID", post.PostID, "error", err)
			return nil, err
		}

		if masterMakerID == "" {
			newMasterMakerId := generateUniqueID()
			masterMakerID = newMasterMakerId

			queryMasterMakerb := `UPDATE pmdm.post_management_master
							SET master_maker_id = $1
							WHERE post_id = $2;
							`
			_, err = pmr.db.Exec(ctx, queryMasterMakerb, newMasterMakerId, post.PostID)
			if err != nil {
				log.Error(gctx, FailedUpdate, "error", err)
				return nil, err
			}
		}

		query = query.Values(
			post.OfficeID,
			post.PostName,
			post.OfficeName,
			post.GroupId,
			post.CadreID,
			post.CadreName,
			filled_status,
			post_status,
			post.AllowancesAttached,
			post.AllowanceDescription,
			post.CreatedBy,
			time.Now(),
			post.Status,
			time.Now(),
			valid_to,
			post.OrderCaseMark,
			post.OrderDate,
			post.UploadOrderDocName,
			post.EstablishmentRegisterID,
			post.Designation,
			post.PayLevel,
			post.GradePay,
			post.PermanentStatus,
			post.EstablishmentRegisterName,
			post.EmployeeGroup,
			post.SanctionedStrength,
			post.PostID,
			post.ApproveStatus,
			post.ApprovePostID,
			post.NewOfficeID,
			post.NewOfficeName,
			post.DesignationId,
			post.ExchangePostID,
			post.Remarks,
			masterMakerID,
		)
	}
	query = query.Suffix("RETURNING postmanagement_maker_id, office_id, post_id, post_name, office_name, status")

	post, err := dblib.InsertReturning(ctx, pmr.db, query, pgx.RowToAddrOfStructByPos[domain.PostManagementMaster3])
	if err != nil {
		return nil, err
	}

	response = append(response, post)

	return response, nil
}

func (pmr *PostManagementRepository) PostManagementByOfficeIDMDWMaker(gctx *gin.Context, officeID int, raPostID int64) ([]domain.PostManagementMaker, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select(QOfficeID, "pm.post_name", "kom.office_name", QGroupID, QFilledStatus, QPostID, "pm.designation", QPermanentStatus, QStatus).
		From(PostManagementMasterMaker).
		Join("pmdm.post_mapping_detail pmd on pmd.employee_post_id=pm.post_id and pmd.role_authority=" + strconv.Itoa(int(raPostID))).
		Join("pmdm.kafka_office_master kom ON kom.office_id = pmd.employee_office_id").
		Where(sq.Eq{"pmd.employee_office_id": officeID})
	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaker])
}

func (pmr *PostManagementRepository) ApprovePostManagementMakerQuery(gctx *gin.Context, postIDs []int, approvedBy string) (string, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	log.Debug(ctx, DebugMsg)

	// Create a new batch for selecting post details
	selectBatch := &pgx.Batch{}

	// Iterate over post IDs to fetch details
	for _, postID := range postIDs {
		log.Debug(ctx, ProcessingPostID, "postID", postID)
		selectBatch.Queue(`SELECT 
                new_office_id, new_office_name, status, office_id, 
                cadre_id, cadre_name, designation, designation_id, 
                group_id, employee_group, pay_level, grade_pay, post_name, post_status
            FROM 
                pmdm.post_management_master_maker 
            WHERE 
                post_id = $1 AND approve_status = 'Pending'`, postID)
	}

	br := pmr.db.SendBatch(ctx, selectBatch)
	defer br.Close()

	// Prepare a new batch for updates
	updateBatch := &pgx.Batch{}

	for _, postID := range postIDs {
		// Nullable fields
		var (
			newOfficeID, officeID, cadreID, designationID, groupID, payLevel, gradePay         *int
			newOfficeName, status, cadreName, designation, employeeGroup, postName, postStatus *string
		)

		err := br.QueryRow().Scan(
			&newOfficeID, &newOfficeName, &status, &officeID,
			&cadreID, &cadreName, &designation, &designationID,
			&groupID, &employeeGroup, &payLevel, &gradePay, &postName, &postStatus,
		)
		if err != nil {
			log.Error(ctx, "Failed to fetch post details from maker table", "postID", postID, "error", err)
			return "", fmt.Errorf("error fetching post details for postID %d: %w", postID, err)
		}

		// Safely dereference pointers or assign default values
		finalNewOfficeID := defaultInt(newOfficeID)
		finalOfficeID := defaultInt(officeID)
		finalStatus := defaultString(status)

		updatedRemarks := fmt.Sprintf("Status: %s, Office ID: %d, New Office ID: %d", finalStatus, finalOfficeID, finalNewOfficeID)

		// Update approval status in the maker table
		updateBatch.Queue(`UPDATE pmdm.post_management_master_maker
            SET approve_status = 'Approved', approved_by = $1, approved_date = $2, remarks = $3 
            WHERE post_id = $4 AND approve_status = 'Pending'`,
			approvedBy, time.Now(), updatedRemarks, postID)

		// Update post_management_master table with new details
		updateBatch.Queue(`UPDATE pmdm.post_management_master
            SET office_id = $1, office_name = $2, cadre_id = $3, cadre_name = $4, 
                designation = $5, designation_id = $6, group_id = $7, employee_group = $8, 
                pay_level = $9, grade_pay = $10, updated_date = $11, remarks = $12, 
                post_name = $13, post_status = $14
            WHERE post_id = $15`,
			finalNewOfficeID, defaultString(newOfficeName), defaultInt(cadreID), defaultString(cadreName),
			defaultString(designation), defaultInt(designationID), defaultInt(groupID), defaultString(employeeGroup),
			defaultInt(payLevel), defaultInt(gradePay), time.Now(), updatedRemarks, defaultString(postName), defaultString(postStatus), postID)

		// Insert if no rows are updated
		updateBatch.Queue(`INSERT INTO pmdm.post_management_master (
                post_id, office_id, office_name, cadre_id, cadre_name, designation, 
                designation_id, group_id, employee_group, pay_level, grade_pay, 
                updated_date, remarks, post_name, post_status)
            SELECT $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
            WHERE NOT EXISTS (
                SELECT 1 FROM pmdm.post_management_master WHERE post_id = $1
            )`,
			postID, finalNewOfficeID, defaultString(newOfficeName), defaultInt(cadreID), defaultString(cadreName),
			defaultString(designation), defaultInt(designationID), defaultInt(groupID), defaultString(employeeGroup),
			defaultInt(payLevel), defaultInt(gradePay), time.Now(), updatedRemarks, defaultString(postName), defaultString(postStatus))
	}

	// Execute the batch of updates
	batchResults := pmr.db.SendBatch(ctx, updateBatch)
	defer batchResults.Close()

	for _, postID := range postIDs {
		_, err := batchResults.Exec()
		if err != nil {
			log.Error(gctx, "Failed to update approval status or post details", "postID", postID, "error", err)
			return "", err
		}
		log.Debug(gctx, "Processed post ID", "postID", postID)
	}

	// Return success message
	successMessage := fmt.Sprintf("Successfully approved %d records", len(postIDs))
	log.Debug(gctx, ReturningSuccessMsg, "message", successMessage)

	return successMessage, nil
}

// Utility functions for handling NULL values
func defaultInt(val *int) int {
	if val == nil {
		return 0
	}
	return *val
}

func defaultString(val *string) string {
	if val == nil {
		return ""
	}
	return *val
}

func (svc *PostManagementRepository) PostManagementWithPendingStatusOfMakerQuery(gctx *gin.Context, PostID string) ([]domain.PostManagementMaker1, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), svc.cfg.GetDuration(DBtimeout))
	defer cancel()

	// Construct the query with the correct where clause using And to combine conditions
	query := dblib.Psql.Select(
		PMOfficeID,
		PMPostName,
		PMOfficeName,
		PMGroupID,
		PMFilledStatus,
		PMPostID,
		PMDestinaton,
		PMPermanentStatus, // Assuming boolean default is false
		"COALESCE(pm.approve_status, '') AS approve_status",
		PMStatus,
		COALESCENewOfficeID,   // Using COALESCE for new_office_id
		COALESCENewOfficeName, // Using COALESCE for new_office_name
		COALESCERemarks,
		"COALESCE(pm.exchange_post_id, 0) AS exchange_post_id",
		"COALESCE(pm.order_date, '1900-01-01') AS order_date", // Default date, adjust as needed
		COALESCEEmployeeGroup,
		PMCradeID,
		COALESCECadreName,
		COALESCEPayLevel,
		// Fields from the old record (main table)
		"COALESCE(old_pm.cadre_id, 0) AS old_cadre_id",
		"COALESCE(old_pm.cadre_name, '') AS old_cadre_name",
		"COALESCE(old_pm.group_id, 0) AS old_group_id",
		"COALESCE(old_pm.designation, '') AS old_designation",
		"COALESCE(old_pm.pay_level, 0) AS old_pay_level",
		"COALESCE(old_pm.grade_pay, 0) AS old_grade_pay",
		"COALESCE(old_pm.status, '') AS old_status",
		"COALESCE(old_pm.employee_group, '') AS old_employee_group",
		"COALESCE(old_pm.post_name, '') AS old_post_name",
	).From("pmdm.post_management_master_maker pm").
		LeftJoin("pmdm.post_management_master old_pm ON old_pm.post_id = pm.post_id").
		Where(sq.And{
			sq.Eq{"pm.approve_post_id": PostID},
			sq.Eq{PMApproveStatus: "Pending"},
			sq.NotEq{QPostID: 0}, // Exclude records where post_id is 0
			sq.NotEq{QStatus: NewPost},
		})
	return dblib.SelectRows(ctx, svc.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaker1])
}

func (pmr *PostManagementRepository) ApprovePostManagementMakerForAbolishPost(gctx *gin.Context, postIDs []int, approvedBy string) (string, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	log.Debug(gctx, "Starting batch processing for approving post management maker records")

	// Create a new batch for selecting post details
	selectBatch := &pgx.Batch{}

	// Iterate over post IDs to fetch details
	for _, postID := range postIDs {
		log.Debug(gctx, ProcessingPostID, "postID", postID)
		selectBatch.Queue(`
            SELECT 
                new_office_id, new_office_name, status, office_id 
            FROM 
                pmdm.post_management_master_maker 
            WHERE 
                post_id = $1 AND approve_status = 'Pending'
        `, postID)
	}

	br := pmr.db.SendBatch(ctx, selectBatch)
	defer br.Close()

	// Prepare a new batch for updates and deletions
	updateBatch := &pgx.Batch{}

	for _, postID := range postIDs {
		var newOfficeID, officeID int
		var newOfficeName, status string

		err := br.QueryRow().Scan(&newOfficeID, &newOfficeName, &status, &officeID)
		if err != nil {
			log.Error(gctx, "Failed to fetch post details", "postID", postID, "error", err)
			return "", err
		}

		// Construct remarks based on status, office_id, and new_office_id
		updatedRemarks := fmt.Sprintf("Status: %s, Office ID: %d, New Office ID: %d", status, officeID, newOfficeID)

		// Update approval status in the maker table
		updateBatch.Queue(`
            UPDATE pmdm.post_management_master_maker
            SET approve_status = 'Approved', approved_by = $1, approved_date = $2, remarks = $3 
            WHERE post_id = $4 AND approve_status = 'Pending'
        `, approvedBy, time.Now(), updatedRemarks, postID)

		// Delete the record from post_management_master table
		updateBatch.Queue(`
            DELETE FROM pmdm.post_management_master
            WHERE post_id = $1
        `, postID)
	}

	// Execute the batch of updates and deletions
	batchResults := pmr.db.SendBatch(ctx, updateBatch)
	defer batchResults.Close()

	for _, postID := range postIDs {
		_, err := batchResults.Exec()
		if err != nil {
			log.Error(gctx, "Failed to update approval status or delete record", "postID", postID, "error", err)
			return "", err
		}
		log.Debug(gctx, "Updated approval status and deleted record", "postID", postID)
	}

	// Return success message
	successMessage := fmt.Sprintf("Successfully approved and deleted %d records", len(postIDs))
	log.Debug(gctx, ReturningSuccessMsg, "message", successMessage)

	return successMessage, nil
}
func (pmr *PostManagementRepository) PostManagementWithApprovedStatusOfMakerByOfficeID(gctx *gin.Context, officeID int) ([]domain.PostManagementMaker, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	//var postList []domain.PostManagementMaker

	//log.Println("Building SQL query...")
	query := dblib.Psql.Select(
		PMOfficeID,
		PMOfficeName,
		PMGroupID,
		PMFilledStatus,
		PMPostID,
		PMDestinaton,
		PMPermanentStatus, // Boolean field default to false
		PMStatus,
		COALESCECadreName,
		COALESCEEmployeeGroup,
		PMCradeID,
		"COALESCE(pm.approve_status, '') AS approve_status",
		COALESCERemarks,
		COALESCENewOfficeID,
		COALESCENewOfficeName,
		PMPostName).
		From(PostManagementMasterMaker).
		Where(sq.Eq{"office_id": officeID}).
		Where(sq.Eq{"approve_status": "Approved"}).
		OrderBy(QPostID)
	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaker])
}

func (pmr *PostManagementRepository) PostManagementWithPendingCreatePostByOfficeID(gctx *gin.Context, officeID int) ([]domain.PostManagementMaker, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select(
		PMOfficeID,
		PMOfficeName,
		PMGroupID,
		PMFilledStatus,
		PMPostID,
		PMDestinaton,
		PMPermanentStatus,
		PMStatus,
		COALESCECadreName,
		COALESCEEmployeeGroup,
		PMCradeID,
		"COALESCE(pm.approve_status, '') AS approve_status",
		"COALESCE(pm.approve_post_id, '') AS approve_post_id",
		COALESCERemarks,
		COALESCENewOfficeID,
		COALESCENewOfficeName,
		PMPostName,
		COALESCEPayLevel,
		COALESCEDestinationID,
		"COALESCE(pm.order_date, '1900-01-01') AS order_date",
		"COALESCE(pm.order_casemark, '') AS order_casemark",
		"COALESCE(pm.created_by, '') AS created_by",
		"COALESCE(pm.created_date, '1900-01-01') AS created_date",
	).
		From(PostManagementMasterMaker).
		Where(sq.Eq{"office_id": officeID}).
		Where(sq.Eq{"approve_status": "Pending"}).
		Where(sq.Eq{QStatus: NewPost}).
		OrderBy(QPostID)
	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaker])
}

func (svc *PostManagementRepository) PostManagementChangFilledStatusByPostID(gctx *gin.Context, PostID int) (string, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), svc.cfg.GetDuration(DBtimeout))
	defer cancel()

	//log.Println("Starting transaction to change filled status by PostID")

	// Update filled_status, updated_date, and approved_date
	updateQuery := dblib.Psql.Update(PostManagementMaster).
		Set("filled_status", "Filled").
		Set("updated_date", time.Now()).
		Set("approved_date", time.Now()).
		Where(sq.Eq{"post_id": PostID})

	// Execute the update query using the utility function
	result, err := dblib.Update(ctx, svc.db, updateQuery)
	if err != nil {
		log.Error(gctx, "Error updating filled status for PostID:", PostID, "Error:", err)
		return "", err
	}

	rowsAffected := result.RowsAffected()
	if rowsAffected == 0 {
		log.Error(gctx, "No rows were affected by the update query")
		return "", errors.New("no rows were affected by the update query")
	}

	log.Error(gctx, "Update query executed successfully")

	return "Post filled status updated successfully", nil
}

func (ecr *PostManagementRepository) UploadFile(ctx *gin.Context, file multipart.File, objectName, contentType string, size int64) error {
	log.Debug(ctx, "Starting UploadFile repository function...")

	// Check if MinIO client is initialized
	if ecr.MinioClient == nil {
		errMsg := "MinIO client is not initialized"
		log.Error(ctx, errMsg)
		// return &apierrors.AppError{ // Return a pointer to AppError
		//     Message:    errMsg,
		//     StatusCode: "500",
		//     Err:        errors.New(errMsg),
		// }
	}

	log.Debug(ctx, "MinIO client is initialized.")

	// Check if the bucket exists
	bucketName := ecr.cfg.GetString("minio.bucketName")
	log.Debug(ctx, "Bucket name: ", bucketName)

	exists, err := ecr.MinioClient.BucketExists(context.Background(), bucketName)
	if err != nil || !exists {
		log.Error(ctx, "Bucket verification failed: ", err)
		// return &apierrors.AppError{ // Return a pointer to AppError
		//     Message:    "Bucket not found",
		//     StatusCode: "404",
		//     Err:        err,
		// }
	}
	log.Debug(ctx, "Bucket exists and is verified.")

	// Upload the file to MinIO
	log.Debug(ctx, "Uploading file to MinIO with object name: ", objectName)
	_, err = ecr.MinioClient.PutObject(
		context.Background(),
		bucketName,
		objectName,
		file,
		size,
		minio.PutObjectOptions{ContentType: contentType},
	)
	if err != nil {
		log.Error(ctx, "Failed to upload file to MinIO: ", err.Error())
		// return &apierrors.AppError{ // Return a pointer to AppError
		//     Message:    "MinIO upload failed",
		//     StatusCode: "500",
		//     Err:        err,
		// }
	}
	log.Info(ctx, "File uploaded successfully to MinIO: ", objectName)
	return nil
}

//	func (ecr *PostManagementRepository) UploadFile(file multipart.File, objectName, contentType string, size int64) error {
//		fmt.Println("inside upload File repo")
//		if ecr.db == nil {
//			errMsg := "database connection is not initialized"
//			apperror := apierrors.NewAppError(errMsg, "500", errors.New(errMsg))
//			return &apperror
//		}
//		fmt.Println("bucketname before mino call")
//		bucketName := ecr.cfg.GetString("minio.bucketName")
//		fmt.Println("bucketname after minio init", *ecr.MinioClient)
//		if ecr.MinioClient == nil {
//			errMsg := "MinIO client is not initialized"
//			apperror := apierrors.NewAppError(errMsg, "500", errors.New(errMsg)) // Setting HTTP status code to "500" for Internal Server Error
//			return &apperror
//		}
//		_, err := ecr.MinioClient.PutObject(
//			context.Background(),
//			bucketName,
//			objectName,
//			file,
//			size,
//			minio.PutObjectOptions{ContentType: contentType},
//		)
//		//fmt.Println("error minio", err.Error())
//		return err
//	}
func (r *PostManagementRepository) InsertDocument(ctx *gin.Context, doc domain.Document) error {
	// Prepare the SQL query using Squirrel
	psql := sq.StatementBuilder.PlaceholderFormat(sq.Dollar)
	query := psql.Insert("pmdm.document_master_pmdm").
		Columns(
			"office_id", "document_name", "document_type", "document_size",
			"document_file_path", "document_upload_status", "document_uploaded_by",
			"document_uploaded_date").
		Values(
			doc.OfficeID, doc.DocumentName, doc.DocumentType,
			doc.DocumentSize, doc.DocumentFilePath, doc.DocumentUploadStatus,
						doc.DocumentUploadedBy, doc.DocumentUploadedDate).
		Suffix("RETURNING post_id") // Specify to return post_id

	sql, args, err := query.ToSql()
	if err != nil {
		log.Debug(ctx, "Error building SQL query:", err)
		return err
	}

	// Execute the query and scan the result
	var postID int
	err = r.db.QueryRow(ctx, sql, args...).Scan(&postID)
	if err != nil {
		log.Debug(ctx, "Error executing query:", err)
		return err
	}

	// Log the document uploaded
	log.Debug(ctx, "Document uploaded successfully:", doc)

	return nil // Return nil after successful insertion and obtaining post_id
}

// func (ecr *PostManagementRepository) DownloadFile(objectName string) (io.Reader, error) {
// 	bucketName := ecr.cfg.GetString("minio.bucketName")
// 	object, err := ecr.MinioClient.GetObject(context.Background(), bucketName, objectName, minio.GetObjectOptions{})
// 	if err != nil {
// 		return nil, err
// 	}

// 	_, err = object.Stat()
// 	if err != nil {
// 		if minio.ToErrorResponse(err).Code == "NoSuchKey" {
// 			return nil, errors.New("file not found")
// 		}
// 		return nil, err
// 	}

// 	return object, nil
// }

// func (r *PostManagementRepository) GetDocumentsByOfficeID(ctx context.Context, officeID int) ([]domain.Document, error) {
// 	psql := sq.StatementBuilder.PlaceholderFormat(sq.Dollar)
// 	query := psql.Select(
// 		"office_id", "document_name", "document_file_path").
// 		From("pmdm.document_master_pmdm").
// 		Where(sq.Eq{"office_id": officeID})

//		return dblib.SelectRows(ctx, r.db, query, pgx.RowToStructByNameLax[domain.Document])
//	}
func (r *PostManagementRepository) GetDocumentsByOfficeID(ctx context.Context, officeID int) ([]domain.Document, error) {
	psql := sq.StatementBuilder.PlaceholderFormat(sq.Dollar)
	query := psql.Select(
		"office_id", "document_name", "document_file_path").
		From("pmdm.document_master_pmdm").
		Where(sq.Eq{"office_id": officeID})

	documents, err := dblib.SelectRows(ctx, r.db, query, pgx.RowToStructByNameLax[domain.Document])
	if err != nil {
		log.Error(ctx, "Database query failed", "error", err)
		return nil, fmt.Errorf("database query failed: %w", err)
	}

	if len(documents) == 0 {
		log.Warn(ctx, "No documents found for the given office ID", "office_id", officeID)
		return nil, errors.New("no documents found for the given office ID")
	}

	return documents, nil
}

func (de *PostManagementRepository) DownloadFile(objectName string) (io.Reader, error) {
	bucketName := de.cfg.GetString("minio.bucketName")
	object, err := de.MinioClient.GetObject(context.Background(), bucketName, objectName, minio.GetObjectOptions{})
	if err != nil {
		return nil, err
	}

	_, err = object.Stat()
	if err != nil {
		if minio.ToErrorResponse(err).Code == "NoSuchKey" {
			return nil, errors.New("file not found")
		}
		return nil, err
	}

	return object, nil
}

func (pmr *PostManagementRepository) FetchSurplusPostRecordByApproverPostID(gctx *gin.Context, approverPostID string) ([]domain.PostManagementMaker, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	log.Debug(gctx, "Building SQL query...")
	query := dblib.Psql.Select(
		COALESCENewOfficeID,
		COALESCENewOfficeName,
		PMStatus,
		PMOfficeID,
		PMOfficeName,
		PMCradeID,
		COALESCECadreName,
		PMDestinaton,
		COALESCEDestinationID, // Assuming designation_id is a new field
		PMGroupID,
		COALESCEEmployeeGroup,
		COALESCEPayLevel,                         // Assuming pay_level is a numeric field
		"COALESCE(pm.grade_pay, 0) AS grade_pay", // Assuming grade_pay is a numeric field
		PMPostName,
		PMPostID,
		PMPermanentStatus,
		COALESCEPostStatus). // Assuming post_status is a field in the table
		From(PostManagementMasterMaker).
		Where(sq.Eq{QStatus: "post_keptinsurplus"}).
		Where(sq.Eq{"pm.approve_post_id": approverPostID}). // Assuming approve_post_id is the correct field name
		OrderBy(QPostID)

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaker])
}

func (pmr *PostManagementRepository) RestoredSurplusPost(gctx *gin.Context, postIDs []int, updatedBy string) (string, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	log.Debug(gctx, "Starting batch update for restoring post management maker records")

	// Create a new batch
	batch := &pgx.Batch{}

	// Iterate over post IDs and add batch queries
	for _, postID := range postIDs {
		log.Debug(gctx, ProcessingPostID, "postID", postID)

		// Fetch necessary details for the current post ID from maker table
		var newOfficeID, officeID int
		var newOfficeName, status, cadreName, designation, employeeGroup, postName, postStatus string
		var cadreID, designationID, groupID, payLevel, gradePay int
		err := pmr.db.QueryRow(ctx, `
            SELECT 
                new_office_id, new_office_name, status, office_id, 
                cadre_id, cadre_name, designation, designation_id, 
                group_id, employee_group, pay_level, grade_pay, post_name, post_status
            FROM 
                pmdm.post_management_master_maker 
            WHERE 
                post_id = $1 AND approve_status = 'Approved' AND status = 'post_keptinsurplus'
        `, postID).Scan(&newOfficeID, &newOfficeName, &status, &officeID,
			&cadreID, &cadreName, &designation, &designationID,
			&groupID, &employeeGroup, &payLevel, &gradePay, &postName, &postStatus)
		if err != nil {
			if err == pgx.ErrNoRows {
				log.Error(gctx, "No matching post found in maker table", "postID", postID)
				continue
			}
			log.Error(gctx, "Failed to fetch post details from maker table", "postID", postID, "error", err)
			return "", err
		}

		log.Debug(gctx, "Fetched post details", "postID", postID, "newOfficeID", newOfficeID, "newOfficeName", newOfficeName, "status", status, "officeID", officeID, "cadreID", cadreID, "cadreName", cadreName, "designation", designation, "designationID", designationID, "groupID", groupID, "employeeGroup", employeeGroup, "payLevel", payLevel, "gradePay", gradePay, "postName", postName, "postStatus", postStatus)

		// Construct updated remarks
		updatedRemarks := fmt.Sprintf("Post ID %d restored from surplus on %s by %s. Previous Office: %d, New Office: %d (%s)", postID, time.Now().Format(time.RFC3339), updatedBy, officeID, newOfficeID, newOfficeName)

		// Add the update maker table query to the batch
		batch.Queue(`
            UPDATE pmdm.post_management_master_maker
            SET status = 'Restored', updated_by = $1, updated_date = $2, approve_status = 'Approved', remarks = $3
            WHERE post_id = $4 AND approve_status = 'Approved' AND status = 'post_keptinsurplus'
        `, updatedBy, time.Now(), updatedRemarks, postID)

		// Add the update master table query to the batch
		batch.Queue(`
            UPDATE pmdm.post_management_master
            SET
                office_id = $1,
                office_name = $2,
                cadre_id = $3,
                cadre_name = $4,
                designation = $5,
                designation_id = $6,
                group_id = $7,
                employee_group = $8,
                pay_level = $9,
                grade_pay = $10,
                updated_date = $11,
                remarks = $12,
                post_name = $13,
                post_status = $14,
                status = 'Active'
            WHERE
                post_id = $15
        `, newOfficeID, newOfficeName, cadreID, cadreName, designation, designationID,
			groupID, employeeGroup, payLevel, gradePay, time.Now(), updatedRemarks, postName, postStatus, postID)

		// Add the check for rows affected and insert if necessary to the batch
		batch.Queue(`
            INSERT INTO pmdm.post_management_master (
                post_id, office_id, office_name, cadre_id, cadre_name, designation, 
                designation_id, group_id, employee_group, pay_level, grade_pay, 
                updated_date, remarks, post_name, post_status, status
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, 
                $7, $8, $9, $10, $11, 
                $12, $13, $14, $15, 'Active'
            )
            ON CONFLICT (post_id) DO NOTHING
        `, postID, newOfficeID, newOfficeName, cadreID, cadreName, designation,
			designationID, groupID, employeeGroup, payLevel, gradePay, time.Now(), updatedRemarks, postName, postStatus)
	}

	// Send the batch to the database
	batchResults := pmr.db.SendBatch(ctx, batch)
	defer batchResults.Close()

	// Check for errors in the batch results
	for i := 0; i < batch.Len(); i++ {
		ct, err := batchResults.Exec()
		if err != nil {
			log.Error(gctx, "Failed to execute batch query", "error", err)
			return "", err
		}
		log.Debug(gctx, "Batch query executed", "commandTag", ct.String())
	}

	return "Successfully restored posts", nil
}
func (pmr *PostManagementRepository) RejectPostManagementMaker(gctx *gin.Context, postIDs []int, approvedBy string) (string, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	log.Debug(gctx, "Starting batch processing for rejecting post management maker records")

	batch := &pgx.Batch{}

	// Iterate over post IDs
	for _, postID := range postIDs {
		log.Debug(gctx, ProcessingPostID, "postID", postID)

		// Construct remarks for rejection
		updatedRemarks := fmt.Sprintf("Rejected by %s on %s", approvedBy, time.Now().Format(time.RFC3339))

		// Add update approval status in the maker table to the batch
		batch.Queue(`
            UPDATE pmdm.post_management_master_maker
            SET approve_status = 'Rejected', approved_by = $1, approved_date = $2, remarks = $3
            WHERE post_id = $4 AND approve_status = 'Pending'
        `, approvedBy, time.Now(), updatedRemarks, postID)
		log.Debug(gctx, "Queued update approval status in maker table", "postID", postID)
	}

	// Execute the batch
	br := pmr.db.SendBatch(ctx, batch)
	defer br.Close()

	for i := 0; i < batch.Len(); i++ {
		_, err := br.Exec()
		if err != nil {
			log.Error(gctx, "Failed to execute batch operation", "error", err)
			return "", err
		}
	}

	// Return success message
	successMessage := fmt.Sprintf("Successfully processed %d records", len(postIDs))
	log.Debug(gctx, ReturningSuccessMsg, "message", successMessage)

	return successMessage, nil
}

func (pmr *PostManagementRepository) ApprovePostManagementMakerForExchangePost(gctx *gin.Context, postIDs []int, approvedBy string) (string, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	log.Debug(gctx, "Starting batch processing for approving post management maker records")

	// Create a new batch for selecting post details
	selectBatch := &pgx.Batch{}

	// Iterate over post IDs to fetch details
	for _, postID := range postIDs {
		log.Debug(gctx, ProcessingPostID, "postID", postID)
		selectBatch.Queue(`
            SELECT 
                new_office_id, new_office_name, status, office_id, office_name, exchange_post_id
            FROM 
                pmdm.post_management_master_maker 
            WHERE 
                post_id = $1 and approve_status='Pending'
        `, postID)
	}

	br := pmr.db.SendBatch(ctx, selectBatch)
	defer br.Close()

	// Prepare a new batch for updates
	updateBatch := &pgx.Batch{}

	for _, postID := range postIDs {
		var newOfficeID, officeID, exchangePostID int
		var officeName, newOfficeName, status string

		err := br.QueryRow().Scan(&newOfficeID, &newOfficeName, &status, &officeID, &officeName, &exchangePostID)
		if err != nil {
			log.Error(gctx, "Failed to fetch post details", "postID", postID, "error", err)
			return "", err
		}

		// Construct remarks based on the swap
		remarksForPostID := fmt.Sprintf("Post %d swapped with Post %d: new office ID = %d, new office name = %s", postID, exchangePostID, officeID, officeName)
		remarksForExchangePostID := fmt.Sprintf("Post %d swapped with Post %d: new office ID = %d, new office name = %s", exchangePostID, postID, newOfficeID, newOfficeName)

		// Update approval status in the maker table
		updateBatch.Queue(`
            UPDATE pmdm.post_management_master_maker
            SET approve_status = 'Approved', approved_by = $1, approved_date = $2, remarks = $3 
            WHERE post_id = $4 AND approve_status = 'Pending'
        `, approvedBy, time.Now(), remarksForPostID, postID)

		// Swap office_id, office_name and update remarks between post_id and exchange_post_id in the master table
		updateBatch.Queue(`
            UPDATE pmdm.post_management_master
            SET 
                office_id = CASE 
                    WHEN post_id = $1 THEN (SELECT office_id FROM pmdm.post_management_master WHERE post_id = $2)
                    WHEN post_id = $2 THEN (SELECT office_id FROM pmdm.post_management_master WHERE post_id = $1)
                END,
                office_name = CASE 
                    WHEN post_id = $1 THEN (SELECT office_name FROM pmdm.post_management_master WHERE post_id = $2)
                    WHEN post_id = $2 THEN (SELECT office_name FROM pmdm.post_management_master WHERE post_id = $1)
                END,
                remarks = CASE
                    WHEN post_id = $1 THEN $3
                    WHEN post_id = $2 THEN $4
                END
            WHERE post_id IN ($1, $2)
        `, postID, exchangePostID, remarksForPostID, remarksForExchangePostID)
	}

	// Execute the batch of updates
	batchResults := pmr.db.SendBatch(ctx, updateBatch)
	defer batchResults.Close()

	for _, postID := range postIDs {
		_, err := batchResults.Exec()
		if err != nil {
			log.Error(gctx, "Failed to update approval status or swap records", "postID", postID, "error", err)
			return "", err
		}
		log.Debug(gctx, "Updated approval status and swapped records", "postID", postID)
	}

	// Return success message
	successMessage := fmt.Sprintf("Successfully approved and swapped %d records", len(postIDs))
	log.Debug(gctx, ReturningSuccessMsg, "message", successMessage)

	return successMessage, nil
}

func (pmr *PostManagementRepository) ApprovePostManagementMasterWithMaker(gctx *gin.Context, postIDs []int, approvedBy string, approveStatus string, remarks string) (string, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	log.Debug(gctx, "Starting batch processing for approving/rejecting post management maker records")

	// Validate approveStatus
	if approveStatus != "Approved" && approveStatus != "Rejected" {
		return "", fmt.Errorf("invalid approveStatus: %s", approveStatus)
	}

	status := "Active"
	if approveStatus == "Rejected" {
		status = "Inactive"
	}
	currentTime := time.Now()

	batch := &pgx.Batch{}

	for _, postID := range postIDs {
		// Get master_maker_id from the post_management_master table
		var masterMakerID, newPost string
		queryGetMasterMakerID := `
    SELECT COALESCE(master_maker_id, '') AS master_maker_id
    FROM pmdm.post_management_master
    WHERE post_id = $1`
		err := pmr.db.QueryRow(ctx, queryGetMasterMakerID, postID).Scan(&masterMakerID)
		if err != nil {
			log.Error(gctx, "Failed to retrieve master_maker_id", "postID", postID, "error", err)
			return "", err
		}

		if masterMakerID == "" {
			newMasterMakerId := generateUniqueID()
			masterMakerID = newMasterMakerId
			queryMasterMaker := `UPDATE pmdm.post_management_master_maker
							SET master_maker_id = $1
							WHERE post_id = $2;
							`
			_, err = pmr.db.Exec(ctx, queryMasterMaker, newMasterMakerId, postID)
			if err != nil {
				log.Error(gctx, FailedUpdate, "error", err)
				return "", err
			}

			queryMasterMakerb := `UPDATE pmdm.post_management_master
							SET master_maker_id = $1
							WHERE post_id = $2;
							`
			_, err = pmr.db.Exec(ctx, queryMasterMakerb, newMasterMakerId, postID)
			if err != nil {
				log.Error(gctx, FailedUpdate, "error", err)
				return "", err
			}
		}

		postStatus := `
			 SELECT status
    FROM pmdm.post_management_master_maker
    WHERE post_id = $1
    ORDER BY created_date DESC
    LIMIT 1`
		err = pmr.db.QueryRow(ctx, postStatus, postID).Scan(&newPost)
		if err != nil {
			log.Error(gctx, "Failed to retrieve status", "postID", postID, "error", err)
			return "", err
		}

		// Prepare update query for post_management_master_maker table
		queryMaker := `
			UPDATE pmdm.post_management_master_maker 
			SET approve_status = $1, approved_by = $2, approved_date = $3, post_id = $4, remarks = $5,updated_by = $6, updated_date = $7
			WHERE master_maker_id = $8 AND approve_status = 'Pending'`
		batch.Queue(queryMaker, approveStatus, approvedBy, currentTime, postID, remarks, approvedBy, time.Now(), masterMakerID)

		if approveStatus == "Approved" && newPost != NewPost {
			var postName, cadreName, designation, payLevel, gradePay, cadreID, groupID, designationID, orderCaseMark, orderDate, uploadOrderDocName, empGroup, newOfficeID, newOfficeName string
			queryGetMakerFields := `SELECT 
					COALESCE(post_name, '') AS post_name,
					COALESCE(cadre_name, '') AS cadre_name,
					COALESCE(designation, '') AS designation,
					COALESCE(pay_level, 0) AS pay_level,
					COALESCE(grade_pay, 0) AS grade_pay,
					COALESCE(cadre_id, 0) AS cadre_id,
					COALESCE(group_id, 0) AS group_id,
					COALESCE(designation_id, 0) AS designation_id,
					COALESCE(order_casemark, '') AS order_casemark,
					COALESCE(order_date::TEXT, '') AS order_date,
					COALESCE(upload_order_doc_name, '') AS upload_order_doc_name,
					COALESCE(employee_group, '') AS employee_group,
					COALESCE(new_office_id, 0) AS new_office_id,
					COALESCE(new_office_name, '') AS new_office_name
				FROM pmdm.post_management_master_maker
				WHERE master_maker_id = $1
				ORDER BY 
    			created_date DESC 
				LIMIT 1;`

			err := pmr.db.QueryRow(ctx, queryGetMakerFields, masterMakerID).Scan(
				&postName, &cadreName, &designation, &payLevel, &gradePay, &cadreID, &groupID, &designationID,
				&orderCaseMark, &orderDate, &uploadOrderDocName, &empGroup, &newOfficeID, &newOfficeName)
			if err != nil {
				log.Error(gctx, "Failed to retrieve fields from maker table", "masterMakerID", masterMakerID, "error", err)
				return "", err
			}

			// Update query for post_management_master table for approval
			queryMaster := `
				UPDATE pmdm.post_management_master 
				SET status = $1, approved_by = $2, approved_date = $3, remarks = $4,
					post_name = $5, cadre_name = $6, designation = $7, pay_level = $8, grade_pay = $9,
					cadre_id = $10, group_id = $11, designation_id = $12, order_casemark = $13,
					order_date = $14, upload_order_doc_name = $15,updated_by = $16, updated_date = $17, office_id = $18, office_name = $19 
				WHERE post_id = $20`
			batch.Queue(queryMaster, status, approvedBy, currentTime, remarks, postName, cadreName, designation, payLevel, gradePay,
				cadreID, groupID, designationID, orderCaseMark, orderDate, uploadOrderDocName, approvedBy, time.Now(), newOfficeID, newOfficeName, postID)
		} else if approveStatus == "Approved" && newPost == NewPost {
			var postName, cadreName, designation, payLevel, gradePay, cadreID, groupID, designationID, orderCaseMark, orderDate, uploadOrderDocName, empGroup string
			queryGetMakerFields := `SELECT 
					COALESCE(post_name, '') AS post_name,
					COALESCE(cadre_name, '') AS cadre_name,
					COALESCE(designation, '') AS designation,
					COALESCE(pay_level, 0) AS pay_level,
					COALESCE(grade_pay, 0) AS grade_pay,
					COALESCE(cadre_id, 0) AS cadre_id,
					COALESCE(group_id, 0) AS group_id,
					COALESCE(designation_id, 0) AS designation_id,
					COALESCE(order_casemark, '') AS order_casemark,
					COALESCE(order_date::TEXT, '') AS order_date,
					COALESCE(upload_order_doc_name, '') AS upload_order_doc_name,
					COALESCE(employee_group, '') AS employee_group
				FROM pmdm.post_management_master_maker
				WHERE master_maker_id = $1
				ORDER BY 
    			created_date DESC 
				LIMIT 1;`

			err := pmr.db.QueryRow(ctx, queryGetMakerFields, masterMakerID).Scan(
				&postName, &cadreName, &designation, &payLevel, &gradePay, &cadreID, &groupID, &designationID,
				&orderCaseMark, &orderDate, &uploadOrderDocName, &empGroup)
			if err != nil {
				log.Error(gctx, "Failed to retrieve fields from maker table", "masterMakerID", masterMakerID, "error", err)
				return "", err
			}

			// Update query for post_management_master table for approval
			queryMaster := `
				UPDATE pmdm.post_management_master 
				SET status = $1, approved_by = $2, approved_date = $3, remarks = $4,
					post_name = $5, cadre_name = $6, designation = $7, pay_level = $8, grade_pay = $9,
					cadre_id = $10, group_id = $11, designation_id = $12, order_casemark = $13,
					order_date = $14, upload_order_doc_name = $15,updated_by = $16, updated_date = $17 
				WHERE post_id = $18`
			batch.Queue(queryMaster, status, approvedBy, currentTime, remarks, postName, cadreName, designation, payLevel, gradePay,
				cadreID, groupID, designationID, orderCaseMark, orderDate, uploadOrderDocName, approvedBy, time.Now(), postID)
		} else if approveStatus == "Rejected" {
			// Delete query for post_management_master table for rejection
			queryDeleteMaster := `
				DELETE FROM pmdm.post_management_master 
				WHERE post_id = $1`
			batch.Queue(queryDeleteMaster, postID)
		} else {
			log.Debug(gctx, "Queue update for post ID", "postID", postID)
		}

		log.Debug(gctx, "Queued updates for post ID", "postID", postID)
	}

	// Execute the batch
	br := pmr.db.SendBatch(ctx, batch)
	defer br.Close()

	// Check for errors
	for i := 0; i < batch.Len(); i++ {
		if _, err := br.Exec(); err != nil {
			log.Error(gctx, "Failed to execute batch update for approval/rejection", "error", err)
			return "", err
		}
	}

	// Return success message
	successMessage := fmt.Sprintf("Successfully processed %d records", len(postIDs))
	log.Debug(gctx, ReturningSuccessMsg, "message", successMessage)

	return successMessage, nil
}

func (svc *PostManagementRepository) GetPostManagementMasterWithMaker(gctx *gin.Context, PostID string) ([]domain.PostManagementMaker, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), svc.cfg.GetDuration(DBtimeout))
	defer cancel()

	// Initialize squirrel's query builder
	psql := sq.StatementBuilder.PlaceholderFormat(sq.Dollar)

	// Step 1: Query to fetch master_maker_id based on approve_post_id and approve_status
	masterMakerQuery := psql.Select("COALESCE(master_maker_id, '') AS master_maker_id").
		From("pmdm.post_management_master_maker").
		Where(sq.Eq{"approve_post_id": PostID, "approve_status": "Pending", "status": NewPost})

	//Execute the query to get master_maker_id
	var masterMakerIDs []string
	query, args, err := masterMakerQuery.ToSql()
	if err != nil {
		return nil, err
	}
	rows, err := svc.db.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var masterMakerID string
		err := rows.Scan(&masterMakerID)
		if err != nil {
			return nil, err
		}
		masterMakerIDs = append(masterMakerIDs, masterMakerID)
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	// Step 3: Construct query to fetch required fields including post_id from both tables
	queryBuilder := psql.Select(
		QOfficeID,
		"pm.post_name",
		"pm.office_name",
		QGroupID,
		QFilledStatus,
		"pm.designation",
		QPermanentStatus,
		PMApproveStatus,
		QStatus,
		COALESCENewOfficeID,
		COALESCENewOfficeName,
		COALESCERemarks,
		"COALESCE(pm.exchange_post_id, 0) AS exchange_post_id",
		"pmh.post_id",   // Include post_id from the joined table
		"pm.order_date", // Include order_date field
		COALESCEEmployeeGroup,
		PMCradeID,
		COALESCECadreName,
	).
		From(PostManagementMasterMaker).
		Join("pmdm.post_management_master pmh ON pm.master_maker_id = pmh.master_maker_id").
		Where(sq.And{
			sq.Eq{PMApproveStatus: "Pending"},
			sq.Eq{"pm.master_maker_id": masterMakerIDs},
			sq.Eq{QStatus: NewPost}, // Use master_maker_id fetched earlier
		})

	return dblib.SelectRows(ctx, svc.db, queryBuilder, pgx.RowToStructByNameLax[domain.PostManagementMaker])
}

func (pmr *PostManagementRepository) FetchVacantActivePostByOfficeID(gctx *gin.Context, officeID int) ([]domain.PostManagementMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select(
		PMOfficeID,
		PMCradeID,
		PMCradeName,
		PMGroupID,
		PMEmployeeGroup,
		PMPostID,
		PMPostName,
		COALESCEDestinationID,
		PMDestinaton,
		PMFilledStatus,
		COALESCEPostStatus,
		PMStatus,
		"km.employee_id",
		"COALESCE(CONCAT(km.employee_first_name, ' ', km.employee_middle_name, ' ', km.employee_last_name), '') AS employee_name",
	).
		From(PMPostManagementMaster).
		LeftJoin("pmdm.kafka_employee_master km ON km.post_id = pm.post_id AND km.employment_status = 'Active'").
		Where(sq.And{
			sq.Eq{QOfficeID: officeID},
			sq.Eq{QStatus: "Active"},
			sq.Eq{QFilledStatus: "Vacant"},
		})

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaster])
}

func (pmr *PostManagementRepository) FetchAllActivePostByOfficeID(gctx *gin.Context, officeID int) ([]domain.PostManagementMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select(
		PMOfficeID,
		PMCradeID,
		PMCradeName,
		PMGroupID,
		PMEmployeeGroup,
		PMPostID,
		PMPostName,
		COALESCEDestinationID,
		PMDestinaton,
		PMFilledStatus,
		"COALESCE(office_name, '') AS office_name",
		COALESCEPostStatus,
		"km.employee_id",
		"COALESCE(CONCAT(km.employee_first_name, ' ', km.employee_middle_name, ' ', km.employee_last_name), '') AS employee_name",
	).
		From(PMPostManagementMaster).
		LeftJoin("pmdm.kafka_employee_master km ON km.post_id = pm.post_id  AND km.employment_status = 'Active'").
		Where(sq.And{
			sq.Eq{QOfficeID: officeID},
			sq.Eq{QStatus: "Active"},
		})

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaster])
}

func (pmr *PostManagementRepository) PostManagementByEstablishmentRegisterID(gctx *gin.Context, EstablishmentRegisterID int) ([]domain.PostManagementMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()
	log.Debug(gctx, ContextWithTimeout)

	query := dblib.Psql.Select(
		PMOfficeID, // assuming office_id is an integer, use 0 as default
		PMPostName,
		PMOfficeName,
		PMGroupID, // assuming group_id is an integer, use 0 as default
		PMFilledStatus,
		PMPostID, // assuming post_id is an integer, use 0 as default
		PMDestinaton,
		PMPermanentStatus,
		PMStatus,
		PMGroupName,
		PMEmployeeGroup,
		PMCradeID, // assuming cadre_id is an integer, use 0 as default
		PMCradeName,
		"COALESCE(pay_level, 0) AS pay_level", // assuming pay_level is an integer, use 0 as default
		"COALESCE(grade_pay, 0) AS grade_pay", // assuming grade_pay is an integer, use 0 as default
		"COALESCE(designation_id, 0) AS designation_id", // assuming designation_id is an integer, use 0 as default
	).
		From(PMPostManagementMaster).
		//Where(sq.Eq{"office_id": officeID})
		Where(sq.And{sq.Eq{"establishment_register_id": EstablishmentRegisterID}, sq.Eq{"status": "Active"}}) // Adding status condition

	log.Debug(gctx, QueryConstruted)

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaster])
}

func (pmr *PostManagementRepository) GetPostNameMaster(gctx *gin.Context, cadreID int, reqMetaData port.MetaDataRequest) ([]domain.PostManagementMaster5, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()
	log.Debug(gctx, ContextWithTimeout)

	query := dblib.Psql.Select(
		"COALESCE(post_name, '') AS post_name",
		"COALESCE(\"group\", '') AS \"group\"", // group is a reserved word in SQL, so it's quoted
		"COALESCE(cadre, '') AS cadre",
	).
		From("pmdm.post_name_master").
		Where(sq.Eq{"cadre_id": cadreID}). // Adding condition for cadre_id
		OrderBy("post_name").
		Offset(uint64(reqMetaData.Skip * reqMetaData.Limit)).
		Limit(uint64(reqMetaData.Limit))

	log.Debug(gctx, QueryConstruted)

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaster5])
}

func (pmr *PostManagementRepository) ViewEstablishmentRegisterByAuthority(gctx *gin.Context, officeID int, officeType string) ([]domain.PostManagementMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()
	log.Debug(gctx, ContextWithTimeout)

	// Define base query
	query := dblib.Psql.Select(
		PMOfficeID,
		PMPostName,
		"COALESCE(kom.office_name, '') AS office_name", // office name from kafka_office_master
		PMGroupID,
		PMFilledStatus,
		PMPostID,
		PMDestinaton,
		PMPermanentStatus,
		PMStatus,
		PMGroupName,
		COALESCEEmployeeGroup,
		PMCradeID,
		COALESCECadreName,
		COALESCEPayLevel,
		"COALESCE(pm.grade_pay, 0) AS grade_pay",
		COALESCEDestinationID,
	).
		From(PMPostManagementMaster).
		InnerJoin("pmdm.kafka_office_hierarchy koh ON pm.office_id = koh.office_id").
		InnerJoin("pmdm.kafka_office_master kom ON kom.office_id = koh.office_id")
		//Where(sq.Eq{QStatus: "Active"})

	// Modify query based on office type
	switch officeType {
	case "PDN":
		query = query.Where(sq.Eq{"koh.division_office_id": officeID})
	case "RGL":
		query = query.Where(sq.Eq{"koh.region_office_id": officeID})
	case "CRL":
		query = query.Where(sq.Eq{"koh.circle_office_id": officeID})
	default:
		return nil, fmt.Errorf("invalid office type")
	}

	log.Debug(gctx, QueryConstruted)

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaster])
}

// func (pmr *PostManagementRepository) CreatePostManagementMasterQuery(gctx *gin.Context, posts []domain.PostManagementMaster) ([]domain.PostManagementMaster2, error) {
// 	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
// 	defer cancel()

// 	var response []domain.PostManagementMaster2
// 	filled_status := "Vacant"
// 	post_status := "Active"
// 	valid_to := "9999-12-31T15:45:00Z"
// 	status := "Inactive"
// 	approve_status := "Pending"
// 	new_status := NewPost

// 	for _, post := range posts {
// 		// Generate a unique ID to link the records in both tables
// 		masterMakerID := generateUniqueID()

// 		// Prepare the batch for the current post
// 		batch := &pgx.Batch{}

// 		// Insert into postmanagement_master table and get the post ID
// 		queryMaster := `
//             INSERT INTO pmdm.post_management_master (
//                 master_maker_id,
//                 office_id,
//                 post_name,
//                 office_name,
//                 group_id,
//                 cadre_id,
//                 cadre_name,
//                 filled_status,
//                 post_status,
//                 allowances_attached,
//                 allowance_description,
//                 created_by,
//                 created_date,
//                 status,
//                 valid_from,
//                 valid_to,
//                 order_casemark,
//                 order_date,
//                 upload_order_doc_name,
//                 establishment_register_id,
//                 designation,
//                 pay_level,
//                 grade_pay,
//                 permanent_status,
//                 establishment_register_name,
//                 employee_group,
//                 sanctioned_strength,
//                 approve_post_id
//             ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28)
//             RETURNING postmanagement_id, office_id, post_id, post_name, office_name, status`

// 		batch.Queue(queryMaster, masterMakerID, post.OfficeID, post.PostName, post.OfficeName, post.GroupId, post.CadreID, post.CadreName, filled_status, post_status, post.AllowancesAttached, post.AllowanceDescription, post.CreatedBy, time.Now(), status, time.Now(), valid_to, post.OrderCaseMark, post.OrderDate, post.UploadOrderDocName, post.EstablishmentRegisterID, post.Designation, post.PayLevel, post.GradePay, post.PermanentStatus, post.EstablishmentRegisterName, post.EmployeeGroup, post.SanctionedStrength, post.ApprovePostID)

// 		// Prepare insert into postmanagement_master_maker table
// 		queryMaker := `
//             INSERT INTO pmdm.post_management_master_maker (
//                 master_maker_id,
//                 office_id,
//                 post_name,
//                 office_name,
//                 group_id,
//                 cadre_id,
//                 cadre_name,
//                 filled_status,
//                 post_status,
//                 allowances_attached,
//                 allowance_description,
//                 created_by,
//                 created_date,
//                 approve_status,
//                 valid_from,
//                 valid_to,
//                 order_casemark,
//                 order_date,
//                 upload_order_doc_name,
//                 establishment_register_id,
//                 designation,
//                 pay_level,
//                 grade_pay,
//                 permanent_status,
//                 establishment_register_name,
//                 employee_group,
//                 sanctioned_strength,
//                 approve_post_id,
//                 post_id,
//                 status,
//                 remarks
//             ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31)`

// 		remarks := fmt.Sprintf("Request for newly post at %s by %s", time.Now().Format("2006-01-02"), post.ApprovePostID)

// 		// Queue the maker insert, post_id will be retrieved later from the master insert result
// 		batch.Queue(queryMaker, masterMakerID, post.OfficeID, post.PostName, post.OfficeName, post.GroupId, post.CadreID, post.CadreName, filled_status, post_status, post.AllowancesAttached, post.AllowanceDescription, post.CreatedBy, time.Now(), approve_status, time.Now(), valid_to, post.OrderCaseMark, post.OrderDate, post.UploadOrderDocName, post.EstablishmentRegisterID, post.Designation, post.PayLevel, post.GradePay, post.PermanentStatus, post.EstablishmentRegisterName, post.EmployeeGroup, post.SanctionedStrength, post.ApprovePostID, post.PostID, new_status, remarks)

// 		// Execute the batch
// 		br := pmr.db.SendBatch(ctx, batch)
// 		defer br.Close()

// 		// Retrieve the results of the batch execution
// 		masterPost := new(domain.PostManagementMaster2)
// 		err := br.QueryRow().Scan(&masterPost.PostManagementID, &masterPost.OfficeID, &masterPost.PostID, &masterPost.PostName, &masterPost.OfficeName, &masterPost.Status)
// 		if err != nil {
// 			log.Debug(ctx,"Failed to execute batch operation for postmanagement_master", "error", err)
// 			return nil, err
// 		}
// 		response = append(response, *masterPost)

// 		if _, err := br.Exec(); err != nil {
// 			log.Debug(ctx,"Failed to execute insert into postmanagement_master_maker", "error", err)
// 			return nil, err
// 		}
// 	}

// 	return response, nil
// }

func (pmr *PostManagementRepository) CreatePostManagementMasterQuery(gctx *gin.Context, posts []domain.PostManagementMasterNew1) ([]domain.PostManagementMaster2, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	var response []domain.PostManagementMaster2
	filledStatus := "Vacant"
	postStatus := "Active"
	validTo := "9999-12-31T15:45:00Z"
	status := "Inactive"
	approveStatus := "Pending"
	newStatus := NewPost

	for _, post := range posts {
		// Generate a unique ID for master maker
		masterMakerID := generateUniqueID()

		// Insert into post_management_master and retrieve the post_id
		queryMaster := `
    INSERT INTO pmdm.post_management_master (
        master_maker_id,
        office_id,
        post_name,
        office_name,
        group_id,
        cadre_id,
        cadre_name,
        filled_status,
        post_status,
        allowances_attached,
        allowance_description,
        created_by,
        created_date,
        status,
        valid_from,
        valid_to,
        order_casemark,
        order_date,
        upload_order_doc_name,
        establishment_register_id,
        designation,
        pay_level,
        grade_pay,
        permanent_status,
        establishment_register_name,
        employee_group,
        sanctioned_strength,
        approve_post_id,
        employee_type,
		office_type,
		group_name,
		designation_id
    ) VALUES (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, 
        $21, $22, $23, $24, $25, $26, $27, $28,$29, $30, $31, $32
    )
    RETURNING post_id`

		var employeeGroupStr string
		if post.EmployeeGroup.String == "GDS" {
			employeeGroupStr = "GDS"
		} else {
			employeeGroupStr = "DOP"
		}

		var postID int64
		err := pmr.db.QueryRow(ctx, queryMaster,
			masterMakerID,
			post.OfficeID,
			post.PostName,
			post.OfficeName,
			post.GroupId,
			post.CadreID,
			post.CadreName,
			filledStatus,
			postStatus,
			post.AllowancesAttached,
			post.AllowanceDescription,
			post.CreatedBy,
			time.Now(),
			status,
			time.Now(),
			validTo,
			post.OrderCaseMark,
			post.OrderDate,
			post.UploadOrderDocName,
			post.EstablishmentRegisterID,
			post.Designation,
			post.PayLevel,
			post.GradePay,
			post.PermanentStatus,
			post.EstablishmentRegisterName,
			post.EmployeeGroup,
			post.SanctionedStrength,
			post.ApprovePostID,
			employeeGroupStr,
			post.OfficeType,
			post.GroupName,
			post.DesignationId,
		).Scan(&postID)

		if err != nil {
			log.Error(ctx, "Failed to insert into post_management_master", "error", err)
			return nil, err
		}

		// Insert into post_management_master_maker
		queryMaker := `
            INSERT INTO pmdm.post_management_master_maker (
                master_maker_id,
                office_id,
                post_name,
                office_name,
                group_id,
                cadre_id,
                cadre_name,
                filled_status,
                post_status,
                allowances_attached,
                allowance_description,
                created_by,
                created_date,
                approve_status,
                valid_from,
                valid_to,
                order_casemark,
                order_date,
                upload_order_doc_name,
                establishment_register_id,
                designation,
                pay_level,
                grade_pay,
                permanent_status,
                establishment_register_name,
                employee_group,
                sanctioned_strength,
                approve_post_id,
                post_id,
                status,
                remarks,
				designation_id
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32)`

		remarks := fmt.Sprintf("Request for newly post at %s by %s", time.Now().Format("2006-01-02"), post.ApprovePostID.String)

		_, err = pmr.db.Exec(ctx, queryMaker,
			masterMakerID,
			post.OfficeID,
			post.PostName,
			post.OfficeName,
			post.GroupId,
			post.CadreID,
			post.CadreName,
			filledStatus,
			postStatus,
			post.AllowancesAttached,
			post.AllowanceDescription,
			post.CreatedBy,
			time.Now(),
			approveStatus,
			time.Now(),
			validTo,
			post.OrderCaseMark,
			post.OrderDate,
			post.UploadOrderDocName,
			post.EstablishmentRegisterID,
			post.Designation,
			post.PayLevel,
			post.GradePay,
			post.PermanentStatus,
			post.EstablishmentRegisterName,
			post.EmployeeGroup,
			post.SanctionedStrength,
			post.ApprovePostID,
			postID,
			newStatus,
			remarks,
			post.DesignationId,
		)

		if err != nil {
			log.Error(ctx, "Failed to insert into post_management_master_maker", "error", err)
			return nil, err
		}

		masterMakerIDInt, err := strconv.ParseInt(masterMakerID, 10, 64)
		if err != nil {
			log.Error(ctx, "Failed to convert masterMakerID to int64", "error", err)
			return nil, err
		}

		// Append to response
		response = append(response, domain.PostManagementMaster2{
			PostManagementID: masterMakerIDInt, // Assuming masterMakerID is convertible to int64
			OfficeID:         toInt64(post.OfficeID),
			PostID:           postID,
			PostName:         post.PostName.String,
			OfficeName:       post.OfficeName.String,
			Status:           newStatus,
		})
	}

	return response, nil
}

// Convert null.Int32 to int64 with a default value of 0 if null
func toInt64(n null.Int32) int64 {
	if n.Valid {
		return int64(n.Int32)
	}
	return 0
}

func (svc *PostManagementRepository) PostManagementChangeFilledStatus(gctx *gin.Context, filledstatus string, postID int, updatedDate string) (string, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), svc.cfg.GetDuration(DBtimeout))
	defer cancel()

	// Update filled_status, updated_date, and approved_date
	updateQuery := dblib.Psql.Update(PostManagementMaster).
		Set("filled_status", filledstatus).
		Set("updated_date", updatedDate).
		Where(sq.And{
			sq.Eq{"status": "Active"},
			sq.Eq{"post_id": postID},
		})

	_, err := dblib.Update(ctx, svc.db, updateQuery)
	if err != nil {
		log.Error(gctx, "Error updating filled status for PostID:", postID, "Error:", err)
		return "", err
	}

	return "Post filled status updated successfully", nil
}

func (cr *PostManagementRepository) ListPostManagement(gctx *gin.Context, divisionOfficeId int, meta port.MetaDataRequest) ([]*domain.ListManagementMaker, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), cr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select(
		"h.circle_name",
		"h.circle_office_id",
		"h.region_name",
		"h.region_office_id",
		"h.division_name",
		"h.division_office_id",
		"h.sub_division_name",
		"h.sub_division_office_id",
		"h.ho_id",
		"h.ho_name",
		"h.hro_id",
		"h.hro_name",
		"h.so_id",
		"h.so_name",
		"h.sro_id",
		"h.sro_name",
		"h.bo_id",
		"h.bo_name",
		"h.office_name",
		QofficeID,
		PpostID,
		PpostName,
		PGroupID,
		PEmpGrp,
		"p.cadre_id",
		PCadreName,
		PFilledStatus,
		"p.establishment_register_id",
		PDesignation,
		"p.pay_level",
		"p.grade_pay",
		"p.updated_date",
	).
		From("pmdm.kafka_office_hierarchy h").
		Join("pmdm.post_management_master p ON h.office_id = p.office_id").
		Where(sq.And{
			sq.Eq{PPostStatus: "Active"},
			sq.Eq{"p.status": "Active"},
			sq.Expr("current_date <= p.valid_to"),
			sq.Expr("current_date >= p.valid_from"),
			sq.Eq{PEmpGrp: "GDS"},
			sq.Eq{"h.division_office_id": divisionOfficeId},
		}).OrderBy(QofficeID).
		Offset(uint64(meta.Skip * meta.Limit)).
		Limit(uint64(meta.Limit))

	return dblib.SelectRows(ctx, cr.db, query, pgx.RowToAddrOfStructByNameLax[domain.ListManagementMaker])
}
func (ap *PostManagementRepository) ListAvailablePosts(gctx *gin.Context, officeID int, meta port.MetaDataRequest) ([]*domain.ListAvailablePosts, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), ap.cfg.GetDuration(DBtimeout))
	defer cancel()
	query := dblib.Psql.Select(
		QofficeID,
		"p.office_name",
		PpostID,
		PpostName,
		PGroupID,
		PCadreName,
		PDesignation,
		PEmpGrp,
		PPostStatus,
		PFilledStatus,
	).
		From("pmdm.post_management_master p").
		Where(sq.And{
			sq.Eq{QofficeID: officeID},
			sq.Eq{PPostStatus: "Active"},
			sq.Or{
				sq.Expr("current_date BETWEEN p.valid_from AND p.valid_to"),
				sq.Expr("p.valid_to IS NULL"),
			},
		}).OrderBy("office_id").
		Offset(uint64(meta.Skip * meta.Limit)).
		Limit(uint64(meta.Limit))

	return dblib.SelectRows(ctx, ap.db, query, pgx.RowToAddrOfStructByNameLax[domain.ListAvailablePosts])
}

func (ap *PostManagementRepository) ListVacantPosts(gctx *gin.Context, officeID int, meta port.MetaDataRequest) ([]*domain.ListAvailablePosts, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), ap.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select(
		QofficeID,
		"p.office_name",
		PpostID,
		PpostName,
		PGroupID,
		PCadreName,
		PDesignation,
		PEmpGrp,
		PPostStatus,
	).
		From("pmdm.post_management_master p").
		Where(sq.And{
			sq.Eq{QofficeID: officeID},
			sq.Or{
				sq.Eq{PFilledStatus: "Vacant"},
				sq.Eq{PFilledStatus: "VACANT"},
			},
			sq.Eq{PPostStatus: "Active"},
			sq.Or{
				sq.Expr("current_date BETWEEN p.valid_from AND p.valid_to"),
				sq.Expr("p.valid_to IS NULL"),
			},
		}).OrderBy("office_id").
		Offset(uint64(meta.Skip * meta.Limit)).
		Limit(uint64(meta.Limit))

	return dblib.SelectRows(ctx, ap.db, query, pgx.RowToAddrOfStructByNameLax[domain.ListAvailablePosts])
}

func (ap *PostManagementRepository) ListGroupMaster(gctx *gin.Context, meta port.MetaDataRequest) ([]*domain.ListGroupMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), ap.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select(
		"group_id",
		"group_name",
	).
		From("pmdm.group_master").OrderBy("group_id").
		Offset(uint64(meta.Skip * meta.Limit)).
		Limit(uint64(meta.Limit))

	return dblib.SelectRows(ctx, ap.db, query, pgx.RowToAddrOfStructByNameLax[domain.ListGroupMaster])
}

func (ap *PostManagementRepository) ListOfficeDetails(gctx *gin.Context, officeID int64, meta port.MetaDataRequest) ([]*domain.ListOfficeDetails, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), ap.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.
		Select(
			"a.cadre_name",
			"a.group_name",
			"a.post_id",
			"a.designation",
			"a.office_id",
			"c.division_office_id",
			"c.circle_office_id",
			"c.region_office_id",
			"b.reporting_office_id",
			"d.reporting_authority AS reporting_authority_post_id",
			"a.cadre_id",
			"a.designation_id",
		).
		From("pmdm.post_management_master a").
		Join("pmdm.kafka_office_master b ON b.office_id = a.office_id").
		Join("pmdm.kafka_office_hierarchy c ON c.office_id = a.office_id").
		Join("pmdm.post_mapping_detail d ON d.employee_post_id = a.post_id").
		Where(sq.And{
			sq.Eq{"a.office_id": officeID},
			sq.Expr("a.post_id NOT IN (SELECT post_id FROM pmdm.kafka_employee_master where employement_status = 'Active')"),
		}).
		OrderBy("a.office_id").
		Offset(uint64(meta.Skip * meta.Limit)).
		Limit(uint64(meta.Limit))

	result, err := dblib.SelectRows(ctx, ap.db, query, pgx.RowToAddrOfStructByNameLax[domain.ListOfficeDetails])
	if err != nil {
		log.Error(gctx, "Failed to execute query", "error", err)
		return nil, err
	}

	return result, nil
}

func (ap *PostManagementRepository) ListGroupCadre(gctx *gin.Context, groupId int64, meta port.MetaDataRequest) ([]*domain.ListGroupCadre, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), ap.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.
		Select(
			"c.cadre_id",
			"c.cadre_name",
			"c.pay_level",
			"c.grade_pay",
			"g.group_id",
			"g.group_name",
		).
		From("pmdm.cadre_master c").
		Join("pmdm.group_master g ON c.cadre_id::text = ANY(string_to_array(g.cadre_id, ','))").
		Where(sq.Eq{"g.group_id": groupId}).OrderBy("group_id").
		Offset(uint64(meta.Skip * meta.Limit)).
		Limit(uint64(meta.Limit))

	return dblib.SelectRows(ctx, ap.db, query, pgx.RowToAddrOfStructByNameLax[domain.ListGroupCadre])
}
func (ap *PostManagementRepository) CreatePost(ctx *gin.Context, req *domain.CreatePostRequest) ([]int64, error) {
	dbCtx, cancel := context.WithTimeout(ctx.Request.Context(), ap.cfg.GetDuration(DBtimeout))
	defer cancel()

	// Insert into post_management_master and RETURN post_id (not postmanagement_id)
	insertPostQuery := `
INSERT INTO pmdm.post_management_master (
    office_id, post_name, office_name, group_id, cadre_name,
    filled_status, post_status, allowances_attached, allowance_description,
    created_by, created_date, approved_by, approved_date, updated_by, updated_date,
    status, remarks, valid_from, valid_to,
    order_casemark, order_date, upload_order_doc_name,
    establishment_register_id, designation, pay_level, grade_pay,
    permanent_status, establishment_register_name, employee_group,
    sanctioned_strength, group_name, cadre_id, designation_id,
    approve_post_id, master_maker_id, admin_office_id,
    employee_type, office_type, login_id, office_supervisor
)
VALUES (
    $1, $2, $3, $4, $5,
    'Vacant', 'Active', FALSE, NULL,
    $6, CURRENT_DATE, $6, CURRENT_DATE, $6, CURRENT_DATE,
    'Active', NULL, CURRENT_DATE, '9999-12-31',
    NULL, NULL, NULL,
    $1, $7, $8, $9,
    TRUE, $3, $10,
    1, $10, $11, $12,
    NULL, NULL, $13, $14, $15, NULL, NULL
)
RETURNING post_id;
`

	// Insert into post_mapping_detail with only employee_post_id
	insertPostMappingQuery := `
INSERT INTO pmdm.post_mapping_detail (
	employee_post_id, gds_leave_sanc_authority_1, gds_leave_sanc_authority_2, reporting_authority, 
	apar_reporting_authority, apar_review_authority, apar_controlling_authority, apar_appellate_authority, 
	service_book_approve_authority1, leave_sanc_authority_1, leave_sanc_authority_2, leave_sanc_authority_3, 
	updated_date, updated_by, pay_approve_authority1, pay_approve_authority2, leave_fwd_authority1, 
	leave_fwd_authority2, pay_fwd_authority1, pay_fwd_authority2, appointing_authority, disciplinary_authority, 
	ddo_authority, admin_authority, pension_sanctioning_authority, pension_authorising_authority, 
	service_book_approve_authority2, role_authority, employee_office_id, service_book_foward_authority1, 
	service_book_foward_authority2, vigilence_maker_authority, admin_office, apar_custodian_authority
)
VALUES (
    $1, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL,
    NULL, NULL, NULL, NULL
);
`

	var ids []int64
	for i := 0; i < req.NumberOfPosts; i++ {
		var postID int64
		err := ap.db.QueryRow(dbCtx, insertPostQuery,
			req.OfficeID, req.PostName, req.OfficeName, req.GroupID, req.CadreName,
			req.CreatedBy, req.Designation, req.PayLevel, req.GradePay,
			req.EmployeeGroup, req.CadreID, req.DesignationID,
			req.AdminOfficeID, req.EmployeeType, req.OfficeType,
		).Scan(&postID)

		if err != nil {
			return nil, err
		}

		// Insert into post_mapping_detail with the generated post_id
		_, err = ap.db.Exec(dbCtx, insertPostMappingQuery, postID)
		if err != nil {
			return nil, err
		}

		ids = append(ids, postID)
	}

	return ids, nil
}

// func (ap *PostManagementRepository) CreatePost(ctx *gin.Context, req *domain.CreatePostRequest) ([]int, error) {
// 	dbCtx, cancel := context.WithTimeout(ctx.Request.Context(), ap.cfg.GetDuration(DBtimeout))
// 	defer cancel()

// 	query := `INSERT INTO pmdm.post_management_master (
// 		office_id, post_name, office_name, group_id, cadre_name,
// 		filled_status, post_status, allowances_attached, allowance_description,
// 		created_by, created_date, approved_by, approved_date, updated_by, updated_date,
// 		status, remarks, valid_from, valid_to,
// 		order_casemark, order_date, upload_order_doc_name,
// 		establishment_register_id, designation, pay_level, grade_pay,
// 		permanent_status, establishment_register_name, employee_group,
// 		sanctioned_strength, group_name, cadre_id, designation_id,
// 		approve_post_id, master_maker_id, admin_office_id,
// 		employee_type, office_type, login_id, office_supervisor
// 	)
// 	VALUES (
// 		$1, $2, $3, $4, $5,
// 		'Vacant', 'Active', FALSE, NULL,
// 		$6, CURRENT_DATE, $6, CURRENT_DATE, $6, CURRENT_DATE,
// 		'Active', NULL, CURRENT_DATE, '9999-12-31',
// 		NULL, NULL, NULL,
// 		$1, $7, $8, $9,
// 		TRUE, $3, $10,
// 		1, $10, $11, $12,
// 		NULL, NULL, $13, $14, $15, NULL, NULL
// 	)
// 	RETURNING post_id;`

// 	var ids []int
// 	for i := 0; i < req.NumberOfPosts; i++ {
// 		var postID int
// 		err := ap.db.QueryRow(dbCtx, query,
// 			req.OfficeID, req.PostName, req.OfficeName, req.GroupID, req.CadreName,
// 			req.CreatedBy, req.Designation, req.PayLevel, req.GradePay,
// 			req.EmployeeGroup, req.CadreID, req.DesignationID,
// 			req.AdminOfficeID, req.EmployeeType, req.OfficeType,
// 		).Scan(&postID)
// 		if err != nil {
// 			return nil, err
// 		}
// 		ids = append(ids, postID)
// 	}
// 	return ids, nil
// }

func (ap *PostManagementRepository) UpdatePost(gctx *gin.Context, req *domain.UpdatePostRequest) (string, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), ap.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := `UPDATE pmdm.post_management_master SET office_id = $2, office_name = $3, admin_office_id = $4, office_type = $5 where post_id = $1`

	_, err := ap.db.Exec(ctx, query,
		req.PostId,
		req.OfficeID,
		req.OfficeName,
		req.AdminOfficeID,
		req.OfficeType,
	)
	if err != nil {
		log.Error(gctx, "Failed to update post", "error", err)
		return "", err
	}

	return "Post updated successfully", nil
}

func (ap *PostManagementRepository) PostManagementByOfficeID(gctx *gin.Context, officeID int64) ([]domain.VacantPost, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), ap.cfg.GetDuration(DBtimeout))
	defer cancel()

	queryBuilder := dblib.Psql.
		Select(
			"post_id",
		).
		From("pmdm.post_management_master").Where(sq.And{
		sq.Eq{"office_id": officeID},
		sq.Eq{"filled_status": "Vacant"},
	})
	return dblib.SelectRows(ctx, ap.db, queryBuilder, pgx.RowToStructByNameLax[domain.VacantPost])
}

func (ap *PostManagementRepository) PostManagementByOfficeIDVacant(gctx *gin.Context, postIds []domain.VacantPost, meta port.MetaDataRequest) ([]domain.VacantPost, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), 10*time.Minute)
	defer cancel()

	var vacantPosts []domain.VacantPost

	query := `
SELECT EXISTS (
	SELECT 1 
	FROM pmdm.kafka_employee_master 
	WHERE post_id = $1 and kafka_employee_master.employment_status = 'Active'
)
`

	for _, post := range postIds {
		var exists bool
		err := ap.db.QueryRow(ctx, query, post.PostID).Scan(&exists)
		if err != nil {
			return nil, fmt.Errorf("failed to check existence for post_id %d: %w", post.PostID, err)
		}
		if !exists {
			vacantPosts = append(vacantPosts, post)
		}
	}

	return vacantPosts, nil
}

func (r *PostManagementRepository) GetPostDetailsByPostID(gctx *gin.Context, postID int64, skip int64, limit int64) ([]domain.PostDetails, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), r.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := `WITH base_post AS (
    SELECT 
        pm.post_id,
        pm.cadre_name,
        pm.office_id,
        pm.is_head_of_the_office,
        om.office_type_code
    FROM 
        pmdm.post_management_master pm
    JOIN 
        pmdm.kafka_office_master om ON pm.office_id = om.office_id
    WHERE 
        pm.post_id = $1
),

reporting_office_id AS (
    SELECT kom.reporting_office_id
    FROM pmdm.kafka_office_master kom
    JOIN base_post bp ON kom.office_id = bp.office_id
    WHERE bp.is_head_of_the_office = TRUE
),

case_self_is_hoo AS (
    SELECT *
    FROM pmdm.post_management_master
    WHERE office_id = (SELECT reporting_office_id FROM reporting_office_id)
      AND is_head_of_the_office = TRUE
      AND cadre_name IN (
          SELECT cadre_name
          FROM pmdm.cadre_order
          WHERE cadre_order < (
              SELECT cadre_order
              FROM pmdm.cadre_order
              WHERE cadre_name = (SELECT cadre_name FROM base_post)
          )
      )
),

division_office AS (
    SELECT koh.division_office_id
    FROM pmdm.kafka_office_hierarchy koh
    JOIN base_post bp ON koh.office_id = bp.office_id
),
target_cadres AS (
    SELECT cadre_name
    FROM pmdm.cadre_order
    WHERE cadre_order < (
        SELECT cadre_order
        FROM pmdm.cadre_order
        WHERE cadre_name = (SELECT cadre_name FROM base_post)
    )
),
case_division_posts AS (
    SELECT *
    FROM pmdm.post_management_master
    WHERE office_id = (SELECT division_office_id FROM division_office)
      AND cadre_name IN (SELECT cadre_name FROM target_cadres)
),
case_hpo_posts AS (
    SELECT * 
    FROM pmdm.post_management_master
    WHERE 
        (
            (office_id = (SELECT office_id FROM base_post) AND is_head_of_the_office = TRUE)
            OR 
            (office_id = (SELECT division_office_id FROM division_office))
        )
      AND cadre_name IN (SELECT cadre_name FROM target_cadres)
),
case_other_office_posts AS (
    SELECT * 
    FROM pmdm.post_management_master
    WHERE 
        office_id = (SELECT office_id FROM base_post)
        AND is_head_of_the_office = TRUE
        AND cadre_name IN (SELECT cadre_name FROM target_cadres)
),

final_posts AS (
    SELECT * FROM case_self_is_hoo
    WHERE EXISTS (SELECT 1 FROM base_post WHERE is_head_of_the_office = TRUE)

    UNION

    SELECT * FROM case_division_posts
    WHERE (SELECT office_type_code FROM base_post) IN ('SPO', 'SRO', 'TMO', 'BPC', 'SPC')
      AND (SELECT is_head_of_the_office FROM base_post) = FALSE

    UNION

    SELECT * FROM case_hpo_posts
    WHERE (SELECT office_type_code FROM base_post) IN ('HPO', 'HRO')
      AND (SELECT is_head_of_the_office FROM base_post) = FALSE

    UNION

    SELECT * FROM case_other_office_posts
    WHERE (SELECT office_type_code FROM base_post) NOT IN ('SPO', 'SRO', 'TMO', 'BPC', 'SPC', 'HPO', 'HRO')
      AND (SELECT is_head_of_the_office FROM base_post) = FALSE
)

SELECT 
    pmm.office_id,
    pmm.office_name,
    pmm.post_id,
    pmm.post_name,
    pmm.group_id,
    pmm.group_name,
    pmm.cadre_id,
    pmm.cadre_name,
    pmm.designation,
    pmm.designation_id,
    pmm.is_head_of_the_office,
    COALESCE(kem.employee_id, 0) AS employee_id,
    COALESCE(
        NULLIF(TRIM(CONCAT(
            kem.employee_first_name, ' ',
            kem.employee_middle_name, ' ',
            kem.employee_last_name
        )), ''),
        'Vacant'
    ) AS employee_name
FROM 
    final_posts pmm
LEFT JOIN 
    pmdm.kafka_employee_master kem 
    ON pmm.post_id = kem.post_id 
    AND kem.employment_status = 'Active'
OFFSET $2 LIMIT $3;`

	rows, err := r.db.Query(ctx, query, postID, skip, limit)
	if err != nil {
		return nil, fmt.Errorf("query execution failed: %w", err)
	}
	defer rows.Close()

	var results []domain.PostDetails
	for rows.Next() {
		var pd domain.PostDetails
		err := rows.Scan(
			&pd.OfficeID,
			&pd.OfficeName,
			&pd.PostID,
			&pd.PostName,
			&pd.GroupID,
			&pd.GroupName,
			&pd.CadreID,
			&pd.CadreName,
			&pd.Designation,
			&pd.DesignationID,
			&pd.IsHeadOfTheOffice,
			&pd.EmployeeID,
			&pd.EmployeeName,
		)
		if err != nil {
			return nil, fmt.Errorf("row scan error: %w", err)
		}
		results = append(results, pd)
	}

	return results, nil
}

func (r *PostManagementRepository) GetCLGrantingPosts(gctx *gin.Context, postID int64, skip int64, limit int64) ([]domain.PostDetails, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), r.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := `
WITH base_post AS (
    SELECT 
        pm.post_id, 
        pm.cadre_name, 
        pm.office_id,
        pm.is_head_of_the_office
    FROM 
        pmdm.post_management_master pm
    WHERE 
        pm.post_id = $1
),
reporting_office AS (
    SELECT kom.reporting_office_id
    FROM pmdm.kafka_office_master kom
    JOIN base_post bp ON kom.office_id = bp.office_id
    WHERE bp.is_head_of_the_office = TRUE
),
target_office AS (
    SELECT 
        -- Use reporting office if HoO, else use own office
        COALESCE((
            SELECT reporting_office_id FROM reporting_office
        ), (
            SELECT office_id FROM base_post
        )) AS office_id
),
target_cadres AS (
    SELECT co.cadre_name
    FROM pmdm.cadre_order co
    WHERE co.cadre_order < (
        SELECT co2.cadre_order
        FROM pmdm.cadre_order co2
        WHERE co2.cadre_name = (SELECT cadre_name FROM base_post)
    )
),
filtered_posts AS (
    SELECT *
    FROM pmdm.post_management_master
    WHERE cadre_name IN (SELECT cadre_name FROM target_cadres)
      AND office_id = (SELECT office_id FROM target_office)
)

SELECT 
    pmm.office_id,
    pmm.office_name,
    pmm.post_id,
    pmm.post_name,
    pmm.group_id,
    pmm.group_name,
    pmm.cadre_id,
    pmm.cadre_name,
    pmm.designation,
    pmm.designation_id,
    pmm.is_head_of_the_office,
    COALESCE(kem.employee_id, 0) AS employee_id,
    COALESCE(
        NULLIF(TRIM(CONCAT(
            kem.employee_first_name, ' ',
            kem.employee_middle_name, ' ',
            kem.employee_last_name
        )), ''),
        'Vacant'
    ) AS employee_name
FROM 
    filtered_posts pmm
LEFT JOIN 
    pmdm.kafka_employee_master kem 
    ON pmm.post_id = kem.post_id 
    AND kem.employment_status = 'Active'
LIMIT $2 OFFSET $3;
`

	rows, err := r.db.Query(ctx, query, postID, limit, skip)
	if err != nil {
		return nil, fmt.Errorf("query failed: %w", err)
	}
	defer rows.Close()

	var results []domain.PostDetails
	for rows.Next() {
		var pd domain.PostDetails
		err := rows.Scan(
			&pd.OfficeID,
			&pd.OfficeName,
			&pd.PostID,
			&pd.PostName,
			&pd.GroupID,
			&pd.GroupName,
			&pd.CadreID,
			&pd.CadreName,
			&pd.Designation,
			&pd.DesignationID,
			&pd.IsHeadOfTheOffice,
			&pd.EmployeeID,
			&pd.EmployeeName,
		)
		if err != nil {
			return nil, fmt.Errorf("row scan failed: %w", err)
		}
		results = append(results, pd)
	}

	return results, nil
}

func (r *PostManagementRepository) GetDDOFilteredPosts(gctx *gin.Context, postID int64, skip int64, limit int64) ([]domain.PostDetails, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), r.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := `
WITH base_post AS (
    SELECT post_id, cadre_name, office_id
    FROM pmdm.post_management_master
    WHERE post_id = $1
),
ddo_office AS (
    SELECT ddo_office_id
    FROM pmdm.kafka_office_hierarchy
    WHERE office_id = (SELECT office_id FROM base_post)
),
filtered_posts AS (
    SELECT *
    FROM pmdm.post_management_master
    WHERE 
        office_id = (SELECT ddo_office_id FROM ddo_office)
        AND group_id < 5
)
SELECT 
    pmm.office_id,
    pmm.office_name,
    pmm.post_id,
    pmm.post_name,
    pmm.group_id,
    pmm.group_name,
    pmm.cadre_id,
    pmm.cadre_name,
    pmm.designation,
    pmm.designation_id,
    pmm.is_head_of_the_office,
    COALESCE(kem.employee_id, 0) AS employee_id,
    COALESCE(
        NULLIF(TRIM(CONCAT(
            kem.employee_first_name, ' ',
            kem.employee_middle_name, ' ',
            kem.employee_last_name
        )), ''),
        'Vacant'
    ) AS employee_name
FROM 
    filtered_posts pmm
LEFT JOIN 
    pmdm.kafka_employee_master kem 
    ON pmm.post_id = kem.post_id 
    AND kem.employment_status = 'Active'
LIMIT $2 OFFSET $3;`

	rows, err := r.db.Query(ctx, query, postID, limit, skip)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []domain.PostDetails
	for rows.Next() {
		var post domain.PostDetails
		if err := rows.Scan(
			&post.OfficeID,
			&post.OfficeName,
			&post.PostID,
			&post.PostName,
			&post.GroupID,
			&post.GroupName,
			&post.CadreID,
			&post.CadreName,
			&post.Designation,
			&post.DesignationID,
			&post.IsHeadOfTheOffice,
			&post.EmployeeID,
			&post.EmployeeName,
		); err != nil {
			return nil, err
		}
		results = append(results, post)
	}

	return results, nil
}
func (pmr *PostManagementRepository) FetchAllActivePostByOfficeID2(gctx *gin.Context, officeID int) ([]domain.PostManagementMasterNew, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select(
		"COALESCE(pm.office_id, 0) AS office_id",
		"COALESCE(pm.cadre_id, 0) AS cadre_id",
		"COALESCE(pm.cadre_name, '') AS cadre_name",
		"COALESCE(pm.group_id, 0) AS group_id",
		"COALESCE(pm.group_name, '') AS group_name",
		"COALESCE(pm.employee_group, '') AS employee_group",
		"COALESCE(pm.post_id, 0) AS post_id",
		"COALESCE(pm.post_name, '') AS post_name",
		"COALESCE(pm.designation_id, 0) AS designation_id",
		"COALESCE(pm.designation, '') AS designation",
		`CASE 
        WHEN km.employee_id IS NULL OR km.employment_status != 'Active' 
            THEN 'Vacant' 
        ELSE 'Filled' 
     END AS filled_status`,
		"COALESCE(kom.office_name, '') AS office_name",
		"COALESCE(km.post_status, '') AS post_status",
		"COALESCE(km.employee_id, 0) AS employee_id",
		"COALESCE(CONCAT(km.employee_first_name, ' ', km.employee_middle_name, ' ', km.employee_last_name), '') AS employee_name",
		"COALESCE(pm.is_head_of_the_office, false) AS is_head_of_the_office",
		"COALESCE(pmd.employee_post_id, 0) AS employee_post_id",
		"COALESCE(pmd.leave_sanc_authority_1, 0) AS leave_sanc_authority_1",
		"COALESCE(pmd.leave_sanc_authority_2, 0) AS leave_sanc_authority_2",
		"COALESCE(pmd.pay_approve_authority1, 0) AS pay_approve_authority1",
		"COALESCE(pmd.appointing_authority, 0) AS appointing_authority",
		"COALESCE(pmd.disciplinary_authority, 0) AS disciplinary_authority",
		"COALESCE(pmd.ddo_authority, 0) AS ddo_authority",
		"COALESCE(pmd.employee_office_id, 0) AS employee_office_id",
		"COALESCE(pmd.vigilence_maker_authority, 0) AS vigilence_maker_authority",
		"COALESCE(pm.permanent_status, false) AS permanent_status",
		"COALESCE(pm.allowances_attached, false) AS allowances_attached",
		"COALESCE(pm.allowance_description, '') AS allowance_description",
		"pm.status",
		"pm.created_by, COALESCE(pm.created_date, '1970-01-01'::timestamp) AS created_date, pm.approved_by, COALESCE(pm.approved_date, '1970-01-01'::timestamp) AS approved_date",
		"pm.updated_by, COALESCE(pm.updated_date, '1970-01-01'::timestamp) AS updated_date, COALESCE(pm.valid_from, '1970-01-01'::timestamp) AS valid_from , COALESCE(pm.valid_to, '1970-01-01'::timestamp) AS valid_to, pm.order_casemark",
		"COALESCE(pm.order_date, '1970-01-01'::timestamp) AS order_date",
		"COALESCE(pm.establishment_register_id, 0) AS establishment_register_id",
		"COALESCE(pm.establishment_register_name, '') AS establishment_register_name",
		"COALESCE(pm.remarks, '') AS remarks",
	).
		From("pmdm.post_management_master pm").
		LeftJoin("pmdm.kafka_employee_master km ON km.post_id = pm.post_id AND km.employment_status = 'Active'").
		LeftJoin("pmdm.post_mapping_detail pmd ON pmd.employee_post_id = pm.post_id").
		LeftJoin("pmdm.kafka_office_master kom ON kom.office_id = pm.office_id").
		Where(sq.And{
			sq.Eq{"pm.office_id": officeID},
			sq.Eq{"pm.status": "Active"},
		})
	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMasterNew])
}
func (pmr *PostManagementRepository) FetchVacantActivePostByOfficeID2(gctx *gin.Context, officeID int) ([]domain.PostManagementMasterNew, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select(
		"COALESCE(pm.office_id, 0) AS office_id",
		"COALESCE(kom.office_name, '') AS office_name",
		"COALESCE(pm.cadre_id, 0) AS cadre_id",
		"COALESCE(pm.cadre_name, '') AS cadre_name",
		"COALESCE(pm.group_id, 0) AS group_id",
		"COALESCE(pm.group_name, '') AS group_name",
		"COALESCE(pm.employee_group, '') AS employee_group",
		"COALESCE(pm.post_id, 0) AS post_id",
		"COALESCE(pm.post_name, '') AS post_name",
		"COALESCE(pm.designation_id, 0) AS designation_id",
		"COALESCE(pm.designation, '') AS designation",
		`CASE 
        WHEN km.employee_id IS NULL OR km.employment_status != 'Active' 
            THEN 'Vacant' 
        ELSE 'Filled' 
     END AS filled_status`,
		"COALESCE(km.post_status, '') AS post_status",
		"COALESCE(pm.status, '') AS status",
		"COALESCE(pm.permanent_status, false) AS permanent_status",
		"COALESCE(pm.allowances_attached, false) AS allowances_attached",
		"COALESCE(pm.allowance_description, '') AS allowance_description",
		"pm.created_by",
		"COALESCE(pm.created_date, '1970-01-01'::timestamp) AS created_date",
		"pm.approved_by",
		"COALESCE(pm.approved_date, '1970-01-01'::timestamp) AS approved_date",
		"pm.updated_by",
		"COALESCE(pm.updated_date, '1970-01-01'::timestamp) AS updated_date",
		"COALESCE(pm.valid_from, '1970-01-01'::timestamp) AS valid_from",
		"COALESCE(pm.valid_to, '1970-01-01'::timestamp) AS valid_to",
		"pm.order_casemark",
		"COALESCE(pm.order_date, '1970-01-01'::timestamp) AS order_date",
		"COALESCE(pm.upload_order_doc_name, '') AS upload_order_doc_name",
		"COALESCE(pm.establishment_register_id, 0) AS establishment_register_id",
		"COALESCE(pm.establishment_register_name, '') AS establishment_register_name",
		"COALESCE(pm.remarks, '') AS remarks",
		"COALESCE(km.employee_id, 0) AS employee_id",
		"COALESCE(CONCAT(km.employee_first_name, ' ', km.employee_middle_name, ' ', km.employee_last_name), '') AS employee_name",
		"COALESCE(pm.is_head_of_the_office, false) AS is_head_of_the_office",
		"COALESCE(pmd.employee_post_id, 0) AS employee_post_id",
		"COALESCE(pmd.leave_sanc_authority_1, 0) AS leave_sanc_authority_1",
		"COALESCE(pmd.leave_sanc_authority_2, 0) AS leave_sanc_authority_2",
		"COALESCE(pmd.pay_approve_authority1, 0) AS pay_approve_authority1",
		"COALESCE(pmd.appointing_authority, 0) AS appointing_authority",
		"COALESCE(pmd.disciplinary_authority, 0) AS disciplinary_authority",
		"COALESCE(pmd.ddo_authority, 0) AS ddo_authority",
		"COALESCE(pmd.employee_office_id, 0) AS employee_office_id",
		"COALESCE(pmd.vigilence_maker_authority, 0) AS vigilence_maker_authority",
	).
		From(PMPostManagementMaster).
		LeftJoin("pmdm.kafka_employee_master km ON km.post_id = pm.post_id AND km.employment_status = 'Active'").
		LeftJoin("pmdm.post_mapping_detail pmd ON pmd.employee_post_id = pm.post_id").
		LeftJoin("pmdm.kafka_office_master kom ON kom.office_id = pm.office_id").
		Where(sq.And{
			sq.Eq{QOfficeID: officeID},
			sq.Eq{QStatus: "Active"},
			sq.Expr(`
            CASE 
                WHEN km.employee_id IS NULL OR km.employment_status != 'Active' 
                    THEN 'Vacant' 
                ELSE 'Filled' 
            END = ?`, "Vacant"),
		})

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMasterNew])
}

func (rmr *PostManagementRepository) GetPostDetailsbyPostID(ctx *gin.Context, postid int) ([]domain.PostIDDetails, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.Select("pm.office_id",
		"om.office_name",
		"pm.post_name",
		"pm.post_id",
		"pm.group_id",
		"pm.group_name",
		"pm.cadre_id",
		"pm.cadre_name",
		"pm.designation_id",
		"pm.designation",
		"pm.post_status",
		"kem.employee_id",
		"COALESCE(NULLIF(TRIM(CONCAT(kem.employee_first_name, ' ', kem.employee_middle_name, ' ', kem.employee_last_name)), ''), 'Vacant') AS employee_name",
	).
		From("pmdm.post_management_master pm").
		Join("pmdm.kafka_office_master om ON pm.office_id = om.office_id").
		LeftJoin("pmdm.kafka_employee_master kem ON pm.post_id = kem.post_id AND kem.employment_status = 'Active'").
		Where(sq.Eq{"pm.post_id": postid})

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.PostIDDetails])
}

func (rmr *PostManagementRepository) GetPostsFilledVacantStatus(ctx *gin.Context, officeID int) ([]domain.PostsStatus, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := `WITH input_office AS (
    SELECT
        office_id,
        CASE
            WHEN office_id = circle_office_id THEN 'CIRCLE'
            WHEN office_id = region_office_id THEN 'REGION'
            WHEN office_id = division_office_id THEN 'DIVISION'
            ELSE 'OFFICE'
        END AS office_level,
        circle_office_id,
        region_office_id,
        division_office_id
    FROM pmdm.kafka_office_hierarchy
    WHERE office_id = $1
	)
	SELECT
    ohm.circle_office_id,
    ohm.circle_name,
    ohm.region_office_id,
    ohm.region_name,
    ohm.division_office_id,
    ohm.division_name,
    ohm.office_id,
    ohm.office_name,
    pm.group_id,
    pm.group_name,
    pm.cadre_id,
    pm.cadre_name,
    COALESCE(COUNT(pm.post_id), 0) AS total_posts,
    COALESCE(SUM(
        CASE 
            WHEN kem.employee_id IS NOT NULL AND kem.employment_status = 'Active' THEN 1 
            ELSE 0 
        END
    ), 0) AS total_filled_posts,
    COALESCE(SUM(
        CASE 
            WHEN kem.employee_id IS NULL OR kem.employment_status != 'Active' THEN 1 
            ELSE 0 
        END
    ), 0) AS total_vacant_posts
	FROM 
    pmdm.post_management_master pm
	LEFT JOIN 
    pmdm.kafka_employee_master kem 
    ON pm.post_id = kem.post_id AND kem.employment_status = 'Active'
	LEFT JOIN 
    pmdm.kafka_office_hierarchy ohm 
    ON pm.office_id = ohm.office_id
	CROSS JOIN input_office io
	WHERE 
    (
        (io.office_level = 'CIRCLE' AND ohm.circle_office_id = io.office_id)
        OR (io.office_level = 'REGION' AND ohm.region_office_id = io.office_id)
        OR (io.office_level = 'DIVISION' AND ohm.division_office_id = io.office_id)
        OR (io.office_level = 'OFFICE' AND ohm.office_id = io.office_id)
    )
	GROUP BY
    ohm.circle_office_id,
    ohm.circle_name,
    ohm.region_office_id,
    ohm.region_name,
    ohm.division_office_id,
    ohm.division_name,
    ohm.office_id,
    ohm.office_name,
    pm.group_id,
    pm.group_name,
    pm.cadre_id,
    pm.cadre_name
	ORDER BY
    ohm.circle_name,
    ohm.region_name,
    ohm.division_name,
    ohm.office_name,
    pm.group_name,
    pm.cadre_name;`

	rows, err := rmr.db.Query(gctx, query, officeID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []domain.PostsStatus
	for rows.Next() {
		var row domain.PostsStatus
		if err := rows.Scan(
			&row.CircleOfficeID,
			&row.CircleName,
			&row.RegionOfficeID,
			&row.RegionName,
			&row.DivisionOfficeID,
			&row.DivisionName,
			&row.OfficeID,
			&row.OfficeName,
			&row.GroupID,
			&row.GroupName,
			&row.CadreID,
			&row.CadreName,
			&row.TotalPosts,
			&row.TotalFilledPosts,
			&row.TotalVacantPosts,
		); err != nil {
			return nil, err
		}
		results = append(results, row)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return results, nil

	// return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.PostsStatus])
}
func (rmr *PostManagementRepository) GetPostsCreatedRedeployedAbolished(ctx context.Context, year int, month int) ([]domain.PostsCreatedRedeployedAbolished, error) {
	gctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	query := `
        WITH created_posts AS (
            SELECT
                office_id,
                COUNT(*) AS posts_created
            FROM pmdm.post_management_master
            WHERE EXTRACT(YEAR FROM created_date) = $1
              AND EXTRACT(MONTH FROM created_date) = $2
            GROUP BY office_id
        ),
        redeployed_posts AS (
            SELECT
                redeployment_to_office_id AS office_id,
                COUNT(*) AS posts_redeployed
            FROM pmdm.post_redeployment_log
            WHERE EXTRACT(YEAR FROM redeployment_on) = $1
              AND EXTRACT(MONTH FROM redeployment_on) = $2
            GROUP BY redeployment_to_office_id
        ),
        abolished_posts AS (
            SELECT
                office_id,
                COUNT(*) AS posts_abolished
            FROM pmdm.post_management_master
            WHERE status = 'Abolished'
              AND EXTRACT(YEAR FROM updated_date) = $1
              AND EXTRACT(MONTH FROM updated_date) = $2
            GROUP BY office_id
        )
        SELECT
            oh.circle_office_id AS circle_id,
            oh.circle_name,
            oh.region_office_id AS region_id,
            oh.region_name,
            oh.division_office_id AS division_id,
            oh.division_name,
            oh.office_id,
            COALESCE(cp.posts_created, 0) AS posts_created,
            COALESCE(rp.posts_redeployed, 0) AS posts_redeployed,
            COALESCE(ap.posts_abolished, 0) AS posts_abolished
        FROM pmdm.kafka_office_hierarchy oh
        LEFT JOIN created_posts cp ON oh.office_id = cp.office_id
        LEFT JOIN redeployed_posts rp ON oh.office_id = rp.office_id
        LEFT JOIN abolished_posts ap ON oh.office_id = ap.office_id
        WHERE
            COALESCE(cp.posts_created, 0) > 0
            OR COALESCE(rp.posts_redeployed, 0) > 0
            OR COALESCE(ap.posts_abolished, 0) > 0
        ORDER BY
            oh.circle_name,
            oh.region_name,
            oh.division_name;
    `
	rows, err := rmr.db.Query(gctx, query, year, month)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []domain.PostsCreatedRedeployedAbolished
	for rows.Next() {
		var row domain.PostsCreatedRedeployedAbolished
		if err := rows.Scan(
			&row.CircleOfficeID,
			&row.CircleName,
			&row.RegionOfficeID,
			&row.RegionName,
			&row.DivisionOfficeID,
			&row.DivisionName,
			&row.OfficeID,
			&row.PostsCreated,
			&row.PostsRedeployed,
			&row.PostsAbolished,
		); err != nil {
			return nil, err
		}
		results = append(results, row)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return results, nil
}

func (rmr *PostManagementRepository) GetPostsFilledVacantStatusDetailed(ctx *gin.Context, officeid int) ([]domain.PostsFilledVacantStatusDetailed, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.Select("pm.office_id",
		"om.office_name",
		"pm.post_name",
		"pm.post_id",
		"pm.group_id",
		"pm.group_name",
		"pm.cadre_id",
		"pm.cadre_name",
		"pm.designation_id",
		"pm.designation",
		"pm.post_status",
		"kem.employee_id",
		"COALESCE(NULLIF(TRIM(CONCAT(kem.employee_first_name, ' ', kem.employee_middle_name, ' ', kem.employee_last_name)), ''), 'Vacant') AS employee_name",
	).
		From("pmdm.post_management_master pm").
		Join("pmdm.kafka_office_master om ON pm.office_id = om.office_id").
		LeftJoin("pmdm.kafka_employee_master kem ON pm.post_id = kem.post_id AND kem.employment_status = 'Active'").
		Where(sq.Eq{"pm.office_id": officeid})

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.PostsFilledVacantStatusDetailed])
}

func (rmr *PostManagementRepository) UpdatePostDetailsbyPostIDRepo(ctx *gin.Context, value domain.UpdatePostManagementMaster) ([]domain.UpdatePostManagementMaster, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()
	batch := &pgx.Batch{}

	uquery := dblib.Psql.Update("pmdm.post_management_master")

	uquery = uquery.
		Set("approve_post_id", value.ApprovePostID).
		Set("cadre_id", value.CadreID).
		Set("cadre_name", value.CadreName).
		Set("designation", value.Designation).
		Set("designation_id", value.DesignationID).
		Set("employee_group", value.EmployeeGroup).
		Set("group_id", value.GroupID).
		Set("office_id", value.OfficeID).
		Set("office_name", value.OfficeName).
		Set("order_casemark", value.OrderCaseMark).
		Set("order_date", value.OrderDate).
		Set("post_id", value.PostID).
		Set("post_name", value.PostName).
		Set("remarks", value.Remarks).
		Set("status", value.Status)

	if value.UploadOrderDocName != "" {
		uquery = uquery.Set("upload_order_doc_name", value.UploadOrderDocName)
	}

	if value.CreatedBy != "" {
		uquery = uquery.Set("created_by", value.CreatedBy)
	}

	if value.GradePay != 0 {
		uquery = uquery.Set("grade_pay", value.GradePay)
	}

	if value.PayLevel != 0 {
		uquery = uquery.Set("pay_level", value.PayLevel)
	}

	uquery = uquery.Where(sq.Eq{"post_id": value.PostID})

	if err := dblib.QueueExecRow(batch, uquery); err != nil {
		return nil, err
	}

	if err := rmr.db.SendBatch(gctx, batch).Close(); err != nil {
		log.Error(gctx, "Error while updating post details: %v", err)
		return nil, err
	}

	return []domain.UpdatePostManagementMaster{value}, nil
}

// func (pmr *PostManagementRepository) ExceptionReportOrderCasemark(ctx *gin.Context, officeID int, reqMetadata port.MetaDataRequest) ([]domain.ExceptionReport, error) {
// 	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
// 	defer cancel()

// 	query := `
// 	SELECT
//     	pmm.office_id,
// 		COALESCE(pmm.group_name, '') AS group_name,
// 		COALESCE(pmm.cadre_name, '') AS cadre_name,
// 		pmm.post_id,
// 		COALESCE(pmm.post_name, '') AS post_name,
// 		COALESCE(koh.circle_name, '') AS circle_name,
// 		COALESCE(koh.region_name, '') AS region_name,
// 		COALESCE(koh.division_name, '') AS division_name,
// 		COALESCE(koh.sub_division_name, '') AS sub_division_name,
// 		COALESCE(koh.office_name, '') AS office_name
// FROM
//     pmdm.post_management_master pmm
// JOIN
//     pmdm.kafka_office_hierarchy koh
//     ON pmm.office_id = koh.office_id
// WHERE
//     pmm.status = 'Active'
//     AND (pmm.order_casemark IS NULL OR pmm.order_casemark='')
// 	and pmm.office_id = $1
// 	LIMIT $2 OFFSET $3;
// 	`

// 	rows, err := pmr.db.Query(gctx, query, officeID, reqMetadata.Limit, reqMetadata.Skip)
// 	if err != nil {
// 		return nil, fmt.Errorf("query execution failed: %w", err)
// 	}
// 	defer rows.Close()

// 	var reports []domain.ExceptionReport

// 	for rows.Next() {
// 		var report domain.ExceptionReport
// 		err := rows.Scan(
// 			&report.OfficeID,
// 			&report.GroupName,
// 			&report.CadreName,
// 			&report.PostID,
// 			&report.PostName,
// 			&report.CircleName,
// 			&report.RegionName,
// 			&report.DivisionName,
// 			&report.SubDivisionName,
// 			&report.OfficeName,
// 		)
// 		if err != nil {
// 			return nil, fmt.Errorf("row scan failed: %w", err)
// 		}
// 		reports = append(reports, report)
// 	}

// 	if err := rows.Err(); err != nil {
// 		return nil, fmt.Errorf("row iteration error: %w", err)
// 	}

// 	return reports, nil
// }

func (pmr *PostManagementRepository) ExceptionReportOrderCasemark(
	ctx *gin.Context,
	reqMetadata port.MetaDataRequest,
) ([]domain.ExceptionReport, error) {

	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := `
	SELECT 
		pmm.office_id,
		COALESCE(pmm.group_name, '') AS group_name,
		COALESCE(pmm.cadre_name, '') AS cadre_name,
		pmm.post_id,
		COALESCE(pmm.post_name, '') AS post_name,
		COALESCE(koh.circle_name, '') AS circle_name,
		COALESCE(koh.region_name, '') AS region_name,
		COALESCE(koh.division_name, '') AS division_name,
		COALESCE(koh.sub_division_name, '') AS sub_division_name,
		COALESCE(koh.office_name, '') AS office_name
	FROM 
		pmdm.post_management_master pmm
	JOIN 
		pmdm.kafka_office_hierarchy koh 
		ON pmm.office_id = koh.office_id
	WHERE 
		pmm.status = 'Active'
		AND (pmm.order_casemark IS NULL OR pmm.order_casemark = '')
	LIMIT $1 OFFSET $2;
	`

	rows, err := pmr.db.Query(gctx, query, reqMetadata.Limit, reqMetadata.Skip)
	if err != nil {
		return nil, fmt.Errorf("query execution failed: %w", err)
	}
	defer rows.Close()

	var reports []domain.ExceptionReport

	for rows.Next() {
		var report domain.ExceptionReport
		err := rows.Scan(
			&report.OfficeID,
			&report.GroupName,
			&report.CadreName,
			&report.PostID,
			&report.PostName,
			&report.CircleName,
			&report.RegionName,
			&report.DivisionName,
			&report.SubDivisionName,
			&report.OfficeName,
		)
		if err != nil {
			return nil, fmt.Errorf("row scan failed: %w", err)
		}
		reports = append(reports, report)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("row iteration error: %w", err)
	}

	return reports, nil
}

func (pmr *PostManagementRepository) EstablishmentRegister(ctx *gin.Context, officeID int, reqMetadata port.MetaDataRequest) ([]domain.ExceptionReport, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := `
	SELECT 
    	pmm.office_id,
		COALESCE(pmm.group_name, '') AS group_name,
		COALESCE(pmm.cadre_name, '') AS cadre_name,
		pmm.post_id,
		COALESCE(pmm.post_name, '') AS post_name,
		COALESCE(koh.circle_name, '') AS circle_name,
		COALESCE(koh.region_name, '') AS region_name,
		COALESCE(koh.division_name, '') AS division_name,
		COALESCE(koh.sub_division_name, '') AS sub_division_name,
		COALESCE(koh.office_name, '') AS office_name
FROM 
    pmdm.post_management_master pmm
JOIN 
    pmdm.kafka_office_hierarchy koh 
    ON pmm.office_id = koh.office_id
WHERE 
    pmm.status = 'Active'
     AND (
        pmm.establishment_register_id IS NULL
        OR pmm.establishment_register_id < 1000000000
    )
	and pmm.office_id = $1
	LIMIT $2 OFFSET $3;
	`

	rows, err := pmr.db.Query(gctx, query, officeID, reqMetadata.Limit, reqMetadata.Skip)
	if err != nil {
		return nil, fmt.Errorf("query execution failed: %w", err)
	}
	defer rows.Close()

	var reports []domain.ExceptionReport

	for rows.Next() {
		var report domain.ExceptionReport
		err := rows.Scan(
			&report.OfficeID,
			&report.GroupName,
			&report.CadreName,
			&report.PostID,
			&report.PostName,
			&report.CircleName,
			&report.RegionName,
			&report.DivisionName,
			&report.SubDivisionName,
			&report.OfficeName,
		)
		if err != nil {
			return nil, fmt.Errorf("row scan failed: %w", err)
		}
		reports = append(reports, report)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("row iteration error: %w", err)
	}

	return reports, nil
}

func (pmr *PostManagementRepository) ExceptionReportOfficeName(ctx *gin.Context, officeID int, reqMetadata port.MetaDataRequest) ([]domain.ExceptionReport, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := `
	SELECT 
    	pmm.office_id,
		COALESCE(pmm.group_name, '') AS group_name,
		COALESCE(pmm.cadre_name, '') AS cadre_name,
		pmm.post_id,
		COALESCE(pmm.post_name, '') AS post_name,
		COALESCE(koh.circle_name, '') AS circle_name,
		COALESCE(koh.region_name, '') AS region_name,
		COALESCE(koh.division_name, '') AS division_name,
		COALESCE(koh.sub_division_name, '') AS sub_division_name,
		COALESCE(koh.office_name, '') AS office_name
FROM 
    pmdm.post_management_master pmm
JOIN 
    pmdm.kafka_office_hierarchy koh 
    ON pmm.office_id = koh.office_id
WHERE 
    pmm.status = 'Active'
    AND (pmm.office_name IS NULL OR pmm.office_name='')
	and pmm.office_id = $1
	LIMIT $2 OFFSET $3;
	`

	rows, err := pmr.db.Query(gctx, query, officeID, reqMetadata.Limit, reqMetadata.Skip)
	if err != nil {
		return nil, fmt.Errorf("query execution failed: %w", err)
	}
	defer rows.Close()

	var reports []domain.ExceptionReport

	for rows.Next() {
		var report domain.ExceptionReport
		err := rows.Scan(
			&report.OfficeID,
			&report.GroupName,
			&report.CadreName,
			&report.PostID,
			&report.PostName,
			&report.CircleName,
			&report.RegionName,
			&report.DivisionName,
			&report.SubDivisionName,
			&report.OfficeName,
		)
		if err != nil {
			return nil, fmt.Errorf("row scan failed: %w", err)
		}
		reports = append(reports, report)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("row iteration error: %w", err)
	}

	return reports, nil
}

func (pmr *PostManagementRepository) ExceptionReportCadreName(ctx *gin.Context, officeID int, reqMetadata port.MetaDataRequest) ([]domain.ExceptionReport, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := `
	SELECT 
    	pmm.office_id,
		COALESCE(pmm.group_name, '') AS group_name,
		COALESCE(pmm.cadre_name, '') AS cadre_name,
		pmm.post_id,
		COALESCE(pmm.post_name, '') AS post_name,
		COALESCE(koh.circle_name, '') AS circle_name,
		COALESCE(koh.region_name, '') AS region_name,
		COALESCE(koh.division_name, '') AS division_name,
		COALESCE(koh.sub_division_name, '') AS sub_division_name,
		COALESCE(koh.office_name, '') AS office_name
FROM 
    pmdm.post_management_master pmm
JOIN 
    pmdm.kafka_office_hierarchy koh 
    ON pmm.office_id = koh.office_id
WHERE 
    pmm.status = 'Active'
    AND (pmm.cadre_name IS NULL OR pmm.cadre_name='')
	and pmm.office_id = $1
	LIMIT $2 OFFSET $3;
	`

	rows, err := pmr.db.Query(gctx, query, officeID, reqMetadata.Limit, reqMetadata.Skip)
	if err != nil {
		return nil, fmt.Errorf("query execution failed: %w", err)
	}
	defer rows.Close()

	var reports []domain.ExceptionReport

	for rows.Next() {
		var report domain.ExceptionReport
		err := rows.Scan(
			&report.OfficeID,
			&report.GroupName,
			&report.CadreName,
			&report.PostID,
			&report.PostName,
			&report.CircleName,
			&report.RegionName,
			&report.DivisionName,
			&report.SubDivisionName,
			&report.OfficeName,
		)
		if err != nil {
			return nil, fmt.Errorf("row scan failed: %w", err)
		}
		reports = append(reports, report)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("row iteration error: %w", err)
	}

	return reports, nil
}

func (pmr *PostManagementRepository) ExceptionReportGroupName(ctx *gin.Context, officeID int, reqMetadata port.MetaDataRequest) ([]domain.ExceptionReport, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := `
	SELECT 
    	pmm.office_id,
		COALESCE(pmm.group_name, '') AS group_name,
		COALESCE(pmm.cadre_name, '') AS cadre_name,
		pmm.post_id,
		COALESCE(pmm.post_name, '') AS post_name,
		COALESCE(koh.circle_name, '') AS circle_name,
		COALESCE(koh.region_name, '') AS region_name,
		COALESCE(koh.division_name, '') AS division_name,
		COALESCE(koh.sub_division_name, '') AS sub_division_name,
		COALESCE(koh.office_name, '') AS office_name
FROM 
    pmdm.post_management_master pmm
JOIN 
    pmdm.kafka_office_hierarchy koh 
    ON pmm.office_id = koh.office_id
WHERE 
    pmm.status = 'Active'
    AND (pmm.group_name IS NULL OR pmm.group_name='')
	and pmm.office_id = $1
	LIMIT $2 OFFSET $3;
	`

	rows, err := pmr.db.Query(gctx, query, officeID, reqMetadata.Limit, reqMetadata.Skip)
	if err != nil {
		return nil, fmt.Errorf("query execution failed: %w", err)
	}
	defer rows.Close()

	var reports []domain.ExceptionReport

	for rows.Next() {
		var report domain.ExceptionReport
		err := rows.Scan(
			&report.OfficeID,
			&report.GroupName,
			&report.CadreName,
			&report.PostID,
			&report.PostName,
			&report.CircleName,
			&report.RegionName,
			&report.DivisionName,
			&report.SubDivisionName,
			&report.OfficeName,
		)
		if err != nil {
			return nil, fmt.Errorf("row scan failed: %w", err)
		}
		reports = append(reports, report)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("row iteration error: %w", err)
	}

	return reports, nil
}

func (rmr *PostManagementRepository) GetCadreWiseReports(ctx *gin.Context, officeID int) ([]domain.CadreWiseReports, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.
		Select(
			"pm.cadre_id",
			"pm.cadre_name",
			"COALESCE(COUNT(DISTINCT pm.post_id), 0) AS total_posts",
			`COALESCE(SUM(
            CASE 
                WHEN kem.employee_id IS NOT NULL AND kem.employment_status = 'Active' THEN 1 
                ELSE 0 
            END
        ), 0) AS total_filled_posts`,
			`COALESCE(SUM(
            CASE 
                WHEN kem.employee_id IS NULL OR kem.employment_status != 'Active' THEN 1 
                ELSE 0 
            END
        ), 0) AS total_vacant_posts`,
		).
		From("pmdm.post_management_master pm").
		LeftJoin("pmdm.kafka_employee_master kem ON pm.post_id = kem.post_id").
		LeftJoin("pmdm.kafka_office_hierarchy ohm ON pm.office_id = ohm.office_id").
		Where(sq.Expr(`
        (
            ? = 35320001
            OR ? IN (
                ohm.office_id,
                ohm.circle_office_id,
                ohm.region_office_id,
                ohm.division_office_id,
                ohm.subdivision_office_id
            )
        )
    `, officeID, officeID)).
		Where(sq.Eq{"pm.status": "Active"}).
		GroupBy("pm.cadre_id", "pm.cadre_name").
		OrderBy("pm.cadre_name")

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.CadreWiseReports])
}

func (rmr *PostManagementRepository) GetCadreWiseReports2(ctx *gin.Context, officeID int) ([]domain.CadreWiseReports, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.
		Select(
			"pm.cadre_id",
			"pm.cadre_name",
			"COUNT(DISTINCT pm.post_id) AS total_posts",
			`COALESCE(SUM(
                CASE 
                    WHEN ap.post_id IS NOT NULL THEN 1
                    ELSE 0
                END
            ), 0) AS total_filled_posts`,
			`COALESCE(SUM(
                CASE 
                    WHEN ap.post_id IS NULL THEN 1
                    ELSE 0
                END
            ), 0) AS total_vacant_posts`,
		).
		From("pmdm.post_management_master pm").
		// only marks whether a post has at least one active employee
		LeftJoin(`(
            SELECT DISTINCT post_id
            FROM pmdm.kafka_employee_master
            WHERE employment_status = 'Active'
        ) AS ap ON ap.post_id = pm.post_id`).
		LeftJoin("pmdm.kafka_office_hierarchy ohm ON pm.office_id = ohm.office_id").
		Where(sq.Expr(`
            (
                ? = 35320001
                OR ? IN (
                    ohm.office_id,
                    ohm.circle_office_id,
                    ohm.region_office_id,
                    ohm.division_office_id,
                    ohm.subdivision_office_id
                )
            )
        `, officeID, officeID)).
		Where(sq.Eq{"pm.status": "Active"}).
		GroupBy("pm.cadre_id", "pm.cadre_name").
		OrderBy("pm.cadre_name")

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.CadreWiseReports])
}

func (r *PostManagementRepository) GetCadreSummary(ctx *gin.Context, cadreID int64, search string, meta port.MetaDataRequest) ([]domain.Summary, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 20*time.Second)
	defer cancel()

	var (
		summaries []domain.Summary
		args      []any
	)

	baseQuery := `
	SELECT
		COALESCE(pmm.group_name, '') AS group_name,
  		COALESCE(pmm.cadre_name, '') AS cadre_name,
		COUNT(*) AS total_posts,
		COUNT(CASE WHEN pmm.filled_status = 'Filled' THEN 1 END) AS total_filled_posts,
		COUNT(CASE WHEN pmm.filled_status = 'Vacant' OR pmm.filled_status IS NULL THEN 1 END) AS total_vacant_posts
	FROM pmdm.post_management_master pmm
	JOIN pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id
	WHERE pmm.status = 'Active' AND pmm.cadre_id = $1
	`
	args = append(args, cadreID)
	argIndex := 2

	if search != "" {
		baseQuery += fmt.Sprintf(` AND (
			LOWER(koh.circle_name) LIKE $%d OR
			LOWER(koh.circle_code) LIKE $%d OR
			LOWER(pmm.group_name) LIKE $%d OR
			LOWER(pmm.cadre_name) LIKE $%d)`, argIndex, argIndex, argIndex, argIndex)
		args = append(args, "%"+strings.ToLower(search)+"%")
		argIndex++
	}

	baseQuery += fmt.Sprintf(`
	GROUP BY koh.circle_name, koh.circle_code, koh.circle_office_id, pmm.group_name, pmm.cadre_name, pmm.cadre_id
	ORDER BY koh.circle_name
	LIMIT $%d OFFSET $%d`, argIndex, argIndex+1)

	args = append(args, meta.Limit, meta.Skip)

	rows, err := r.db.Query(gctx, baseQuery, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var s domain.Summary
		if err := rows.Scan(&s.GroupName, &s.CadreName, &s.TotalPosts, &s.TotalFilled, &s.TotalVacant); err != nil {
			return nil, err
		}
		summaries = append(summaries, s)
	}
	return summaries, nil
}

func (r *PostManagementRepository) GetTotalCircleCount(ctx *gin.Context, cadreID int64, search string) (int, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := `
	SELECT COUNT(DISTINCT koh.circle_office_id)
	FROM pmdm.post_management_master pmm
	JOIN pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id
	WHERE pmm.status = 'Active' AND pmm.cadre_id = $1 AND koh.circle_name IS NOT NULL`
	args := []any{cadreID}

	if search != "" {
		query += ` AND (
			LOWER(koh.circle_name) LIKE $2 OR
			LOWER(koh.circle_code) LIKE $2 OR
			LOWER(pmm.group_name) LIKE $2 OR
			LOWER(pmm.cadre_name) LIKE $2)`
		args = append(args, "%"+strings.ToLower(search)+"%")
	}

	var total int
	err := r.db.QueryRow(gctx, query, args...).Scan(&total)
	return total, err
}

func (r *PostManagementRepository) GetDetailList(ctx *gin.Context, cadreID int64, search string) ([]domain.Detail, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := `
	SELECT
		COALESCE(koh.circle_name, '') AS circle_name,
		COALESCE(koh.circle_code, '') AS circle_code,
		COALESCE(koh.circle_office_id, 0) AS circle_office_id,
		COALESCE(koh.office_name, '') AS office_name,
		COALESCE(koh.office_id, 0) AS office_id,
		COALESCE(kom.office_type_code, '') AS office_type_code,
		COALESCE(kom.pincode, 0) AS pincode,
		COALESCE(koh.division_name, '') AS division_name,
		COALESCE(pmm.post_id, 0) AS post_id,
		COALESCE(pmm.post_name, '') AS post_name,
		COALESCE(pmm.designation, '') AS designation,
		COALESCE(pmm.filled_status, '') AS filled_status,
		COALESCE(pmm.post_status, '') AS post_status
	FROM pmdm.post_management_master pmm
	JOIN pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id
	JOIN pmdm.kafka_office_master kom ON koh.office_id = kom.office_id
	WHERE pmm.status = 'Active' AND pmm.cadre_id = $1 AND koh.circle_name IS NOT NULL
	`
	args := []any{cadreID}
	if search != "" {
		query += ` AND (
			LOWER(koh.circle_name) LIKE $2 OR
			LOWER(koh.circle_code) LIKE $2 OR
			LOWER(koh.office_name) LIKE $2)`
		args = append(args, "%"+strings.ToLower(search)+"%")
	}

	query += ` ORDER BY koh.circle_name, koh.office_name`

	rows, err := r.db.Query(gctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var list []domain.Detail
	for rows.Next() {
		var d domain.Detail
		if err := rows.Scan(
			&d.CircleName, &d.CircleCode, &d.CircleOfficeID,
			&d.OfficeName, &d.OfficeID, &d.OfficeTypeCode,
			&d.Pincode, &d.DivisionName, &d.PostID, &d.PostName,
			&d.Designation, &d.FilledStatus, &d.PostStatus,
		); err != nil {
			return nil, err
		}
		list = append(list, d)
	}
	return list, nil
}

func (r PostManagementRepository) GetCircleSummary(ctx *gin.Context, cadreName string, search string, meta port.MetaDataRequest) ([]domain.CircleSummary, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 20*time.Second)
	defer cancel()

	var (
		summaries []domain.CircleSummary
		args      []any
	)

	baseQuery := `
	SELECT 
        COALESCE(koh.circle_name, '') AS circle_name,
    	COALESCE(koh.circle_office_id, 0) AS circle_office_id,
    	COALESCE(pmm.group_name, '') AS group_name,
    	COALESCE(pmm.cadre_name, '') AS cadre_name,
        COUNT(*) as total_posts,
        COUNT(CASE WHEN pmm.filled_status = 'Filled' THEN 1 END) as total_filled_posts,
        COUNT(CASE WHEN pmm.filled_status = 'Vacant' OR pmm.filled_status IS NULL THEN 1 END) as total_vacant_posts
      FROM pmdm.post_management_master pmm
      JOIN pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id
      WHERE pmm.status = 'Active' 
        AND pmm.cadre_name = $1
        AND koh.circle_office_id IS NOT NULL
        AND koh.circle_office_id <> 0
	`
	args = append(args, cadreName)
	argsIndex := 2

	if search != "" {
		baseQuery += fmt.Sprintf(`
		 AND (
          LOWER(koh.circle_name) LIKE LOWER($%d) OR
          LOWER(pmm.group_name) LIKE LOWER($%d) OR
          LOWER(pmm.cadre_name) LIKE LOWER($%d)
        )`, argsIndex, argsIndex, argsIndex)
		args = append(args, "%"+strings.ToLower(search)+"%")
		argsIndex++
	}

	baseQuery += fmt.Sprintf(`
	  GROUP BY koh.circle_name, koh.circle_office_id, pmm.group_name, pmm.cadre_name
      ORDER BY koh.circle_name, pmm.group_name, pmm.cadre_name
      LIMIT $%d OFFSET $%d`, argsIndex, argsIndex+1)

	args = append(args, meta.Limit, meta.Skip)

	rows, err := r.db.Query(gctx, baseQuery, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var s domain.CircleSummary
		if err := rows.Scan(&s.CircleName, &s.CircleOfficeID, &s.GroupName, &s.CadreName, &s.TotalPosts, &s.TotalFilledPosts, &s.TotalVacantPosts); err != nil {
			return nil, err
		}
		summaries = append(summaries, s)
	}
	return summaries, nil
}

func (r *PostManagementRepository) GetTotalCircleCadreCount(ctx *gin.Context, cadreName string, search string) (int, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := `
	 SELECT COUNT(DISTINCT koh.circle_office_id) as total_count
      FROM pmdm.post_management_master pmm
      JOIN pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id
      WHERE pmm.status = 'Active' 
        AND pmm.cadre_name = $1
        AND koh.circle_office_id IS NOT NULL
        AND koh.circle_office_id <> 0
	`
	args := []any{cadreName}
	if search != "" {
		query += `
		 AND (
          LOWER(koh.circle_name) LIKE LOWER($2) OR
          LOWER(pmm.group_name) LIKE LOWER($2) OR
          LOWER(pmm.cadre_name) LIKE LOWER($2)
        )`
		args = append(args, "%"+strings.ToLower(search)+"%")
	}

	var total int
	err := r.db.QueryRow(gctx, query, args...).Scan(&total)
	return total, err
}

func (r *PostManagementRepository) GetCircleDetailList(ctx *gin.Context, cadreName string, circleOfficeID int64, search string) ([]domain.DetailedPost, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := `
	SELECT 
            COALESCE(pmm.postmanagement_id, 0) AS postmanagement_id,
  			COALESCE(pmm.post_id, 0) AS post_id,
  			COALESCE(pmm.post_name, '') AS post_name,
  			COALESCE(pmm.designation, '') AS designation,
  			COALESCE(pmm.filled_status, '') AS filled_status,
  			COALESCE(pmm.post_status, '') AS post_status,
  			COALESCE(pmm.pay_level, 0) AS pay_level,
  			COALESCE(pmm.grade_pay, 0) AS grade_pay,
  			COALESCE(pmm.sanctioned_strength, 0) AS sanctioned_strength,
  			COALESCE(pmm.permanent_status, true) AS permanent_status,
  			COALESCE(pmm.allowances_attached, false) AS allowances_attached,
  			COALESCE(pmm.allowance_description, '') AS allowance_description,
  			COALESCE(pmm.group_name, '') AS group_name,
  			COALESCE(pmm.cadre_name, '') AS cadre_name,
  			COALESCE(pmm.office_name, '') AS post_office_name,
  			COALESCE(pmm.office_id, 0) AS office_id,
  			COALESCE(kom.office_name, '') AS office_name,
  			COALESCE(kom.office_type_code, '') AS office_type_code,
  			COALESCE(kom.pincode, 0) AS pincode,
  			COALESCE(koh.division_name, '') AS division_name,
  			COALESCE(koh.subdivision_name, '') AS subdivision_name,
  			COALESCE(koh.circle_name, '') AS circle_name,
  			COALESCE(koh.region_name, '') AS region_name
        FROM pmdm.post_management_master pmm
        LEFT JOIN pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id
        LEFT JOIN pmdm.kafka_office_master kom ON pmm.office_id = kom.office_id
        WHERE pmm.status = 'Active' 
          AND pmm.cadre_name = $1
          AND koh.circle_office_id = $2
	`
	args := []any{cadreName, circleOfficeID}
	if search != "" {
		paramIdx := len(args) + 1
		query += fmt.Sprintf(`
			AND (
				LOWER(pmm.post_name) LIKE LOWER($%d) OR
				LOWER(pmm.designation) LIKE LOWER($%d) OR
				LOWER(kom.office_name) LIKE LOWER($%d)
			)
		`, paramIdx, paramIdx, paramIdx)

		args = append(args, "%"+strings.ToLower(search)+"%")
	}
	query += ` ORDER BY kom.office_name, pmm.post_name, pmm.designation`

	rows, err := r.db.Query(gctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var list []domain.DetailedPost
	for rows.Next() {
		var d domain.DetailedPost
		if err := rows.Scan(
			&d.PostManagementID, &d.PostID, &d.PostName, &d.Designation, &d.FilledStatus, &d.PostStatus,
			&d.PayLevel, &d.GradePay, &d.SanctionedStrength, &d.PermanentStatus, &d.AllowancesAttached,
			&d.AllowanceDesc, &d.GroupName, &d.CadreName, &d.PostOfficeName, &d.OfficeID, &d.OfficeName,
			&d.OfficeTypeCode, &d.Pincode, &d.DivisionName, &d.SubdivisionName, &d.CircleName, &d.RegionName,
		); err != nil {
			return nil, err
		}
		list = append(list, d)
	}
	return list, nil
}

func (r *PostManagementRepository) GetDivisionSummaries(ctx *gin.Context, cadreName string, regionID int64, search string, meta port.MetaDataRequest) ([]domain.DivisionSummary, int, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	var summaries []domain.DivisionSummary
	var totalCount int

	searchClause := ""
	params := []interface{}{regionID, cadreName}
	countParams := []interface{}{regionID, cadreName}

	if search != "" {
		searchClause = `AND (
			LOWER(koh.division_name) LIKE LOWER($3) OR
			LOWER(koh.region_name) LIKE LOWER($3) OR
			LOWER(pmm.group_name) LIKE LOWER($3) OR
			LOWER(pmm.cadre_name) LIKE LOWER($3)
		)`
		searchTerm := "%" + search + "%"
		params = append(params, searchTerm)
		countParams = append(countParams, searchTerm)
	}

	baseQuery := fmt.Sprintf(`
		SELECT 
    COALESCE(koh.division_name, '') AS division_name,
    COALESCE(koh.division_office_id, 0) AS division_office_id,
    COALESCE(koh.region_name, '') AS region_name,
    COALESCE(koh.region_office_id, 0) AS region_office_id,
    COALESCE(pmm.group_name, '') AS group_name,
    COALESCE(pmm.cadre_name, '') AS cadre_name,
    COUNT(*) AS total_posts,
    COUNT(CASE WHEN pmm.filled_status = 'Filled' THEN 1 END) AS total_filled_posts,
    COUNT(CASE WHEN pmm.filled_status = 'Vacant' OR pmm.filled_status IS NULL THEN 1 END) AS total_vacant_posts
FROM pmdm.post_management_master pmm
JOIN pmdm.kafka_office_hierarchy koh 
    ON pmm.office_id = koh.office_id
    AND koh.region_office_id = $1 
    AND koh.division_office_id IS NOT NULL 
    AND koh.division_office_id <> 0
WHERE pmm.status = 'Active' 
    AND pmm.cadre_name = $2
    %s
GROUP BY 
    koh.division_name, 
    koh.division_office_id, 
    koh.region_name, 
    koh.region_office_id, 
    pmm.group_name, 
    pmm.cadre_name
ORDER BY koh.division_name
LIMIT $%d OFFSET $%d
	`, searchClause, len(params)+1, len(params)+2)

	params = append(params, meta.Limit, meta.Skip)

	rows, err := r.db.Query(gctx, baseQuery, params...)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()

	for rows.Next() {
		var summary domain.DivisionSummary
		if err := rows.Scan(
			&summary.DivisionName,
			&summary.DivisionOfficeID,
			&summary.RegionName,
			&summary.RegionOfficeID,
			&summary.GroupName,
			&summary.CadreName,
			&summary.TotalPosts,
			&summary.TotalFilledPosts,
			&summary.TotalVacantPosts,
		); err != nil {
			return nil, 0, err
		}
		summaries = append(summaries, summary)
	}

	countQuery := fmt.Sprintf(`
		SELECT COUNT(DISTINCT koh.division_office_id)
		FROM pmdm.post_management_master pmm
		JOIN pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id
			AND koh.region_office_id = $1 
			AND koh.division_office_id IS NOT NULL AND koh.division_office_id <> 0
		WHERE pmm.status = 'Active' AND pmm.cadre_name = $2
		%s
	`, searchClause)

	row := r.db.QueryRow(gctx, countQuery, countParams...)
	err = row.Scan(&totalCount)
	if err != nil {
		return nil, 0, err
	}

	return summaries, totalCount, nil
}

func (r *PostManagementRepository) GetDivisionDetail(ctx *gin.Context, cadreName string, regionID int64, divisionID int64, search string) ([]domain.DivisionDetail, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	var details []domain.DivisionDetail
	params := []interface{}{cadreName, regionID, divisionID}
	paramIndex := 4 // Start index for dynamic params

	queryBuilder := strings.Builder{}
	queryBuilder.WriteString(`
		SELECT 
			pmm.postmanagement_id, 
			pmm.post_id, 
			COALESCE(pmm.post_name, '') AS post_name, 
			COALESCE(pmm.designation, '') AS designation,
			COALESCE(pmm.filled_status, '') AS filled_status, 
			COALESCE(pmm.post_status, '') AS post_status, 
			COALESCE(pmm.pay_level, 0) AS pay_level, 
			COALESCE(pmm.grade_pay, 0) AS grade_pay,
			COALESCE(pmm.sanctioned_strength, 0) AS sanctioned_strength, 
			COALESCE(pmm.permanent_status, true) AS permanent_status, 
			COALESCE(pmm.allowances_attached, false) AS allowances_attached,
			COALESCE(pmm.allowance_description, '') AS allowance_description, 
			COALESCE(pmm.group_name, '') AS group_name, 
			COALESCE(pmm.cadre_name, '') AS cadre_name,
			COALESCE(pmm.office_name, '') AS post_office_name, 
			COALESCE(pmm.office_id, 0) AS office_id,
			COALESCE(kom.office_name, '') AS office_name, 
			COALESCE(kom.office_type_code, '') AS office_type_code, 
			COALESCE(kom.pincode, 0) AS pincode,
			COALESCE(koh.division_name, '') AS division_name, 
			COALESCE(koh.subdivision_name, '') AS subdivision_name, 
			COALESCE(koh.circle_name, '') AS circle_name, 
			COALESCE(koh.region_name, '') AS region_name
		FROM pmdm.post_management_master pmm
		LEFT JOIN pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id
		LEFT JOIN pmdm.kafka_office_master kom ON pmm.office_id = kom.office_id
		WHERE pmm.status = 'Active' 
		  AND pmm.cadre_name = $1 
		  AND koh.region_office_id = $2 
		  AND koh.division_office_id = $3
	`)

	if search != "" {
		queryBuilder.WriteString(fmt.Sprintf(`
			AND (
				LOWER(pmm.post_name) LIKE LOWER($%d) OR
				LOWER(pmm.designation) LIKE LOWER($%d) OR
				LOWER(kom.office_name) LIKE LOWER($%d)
			)
		`, paramIndex, paramIndex, paramIndex))
		params = append(params, "%"+strings.ToLower(search)+"%")
		paramIndex++
	}

	queryBuilder.WriteString(" ORDER BY kom.office_name, pmm.post_name, pmm.designation")

	rows, err := r.db.Query(gctx, queryBuilder.String(), params...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var detail domain.DivisionDetail
		if err := rows.Scan(
			&detail.PostManagementID, &detail.PostID, &detail.PostName, &detail.Designation,
			&detail.FilledStatus, &detail.PostStatus, &detail.PayLevel, &detail.GradePay,
			&detail.SanctionedStrength, &detail.PermanentStatus, &detail.AllowancesAttached,
			&detail.AllowanceDesc, &detail.GroupName, &detail.CadreName,
			&detail.PostOfficeName, &detail.OfficeID,
			&detail.OfficeName, &detail.OfficeTypeCode, &detail.Pincode,
			&detail.DivisionName, &detail.SubdivisionName, &detail.CircleName, &detail.RegionName,
		); err != nil {
			return nil, err
		}
		details = append(details, detail)
	}

	return details, nil
}

func (r *PostManagementRepository) GetRegionInfo(ctx *gin.Context, regionID int64) (domain.RegionInfo, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	row := r.db.QueryRow(gctx, `
	SELECT DISTINCT region_name, circle_name, circle_office_id
	FROM pmdm.kafka_office_hierarchy
	WHERE region_office_id = $1
	LIMIT 1
`, regionID)

	var info domain.RegionInfo
	err := row.Scan(&info.RegionName, &info.CircleName, &info.CircleOfficeID)
	if err != nil {
		return domain.RegionInfo{}, err
	}

	return info, nil
}

func (r *PostManagementRepository) GetHierarchySummaries(ctx *gin.Context, cadreName string, parentOfficeID int64, search string, meta port.MetaDataRequest) ([]domain.HierarchySummary, int, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	var summaries []domain.HierarchySummary
	var totalCount int

	params := []interface{}{parentOfficeID, cadreName}
	countParams := []interface{}{parentOfficeID}
	searchClause := ""
	paramIndex := 3

	if search != "" {
		searchClause = fmt.Sprintf(` AND (
			LOWER(kom.office_name) LIKE LOWER($%d) OR
			LOWER(kom.office_type_code) LIKE LOWER($%d) OR
			LOWER(koh.division_name) LIKE LOWER($%d) OR
			LOWER(koh.subdivision_name) LIKE LOWER($%d) OR
			kom.pincode::text LIKE $%d
		)`, paramIndex, paramIndex, paramIndex, paramIndex, paramIndex)
		searchTerm := "%" + search + "%"
		params = append(params, searchTerm)
		countParams = append(countParams, searchTerm)
		paramIndex++
	}

	params = append(params, meta.Limit, meta.Skip)

	baseQuery := fmt.Sprintf(`
		WITH RECURSIVE office_tree AS (
			SELECT office_id, office_id as root_office_id, 0 as level
			FROM pmdm.kafka_office_master 
			WHERE office_id = $1
			UNION ALL
			SELECT kom.office_id, ot.root_office_id, ot.level + 1
			FROM pmdm.kafka_office_master kom
			INNER JOIN office_tree ot ON kom.reporting_office_id = ot.office_id
			WHERE ot.level < 10
		),
		direct_reports AS (
			SELECT DISTINCT office_id FROM pmdm.kafka_office_master WHERE reporting_office_id = $1
		),
		office_with_subordinates AS (
			SELECT dr.office_id as direct_office_id, ot.office_id as subordinate_office_id
			FROM direct_reports dr
			LEFT JOIN office_tree ot ON (ot.root_office_id = dr.office_id OR ot.office_id = dr.office_id)
		)
		SELECT 
			kom.office_id,
			kom.office_name,
			kom.office_type_code,
			kom.pincode,
			kom.reporting_office_id,
			COALESCE(koh.division_name, '') as division_name,
			COALESCE(koh.subdivision_name, '') as subdivision_name,
			COALESCE(koh.circle_name, '') as circle_name,
			COUNT(pmm.postmanagement_id) as total_posts,
			COUNT(CASE WHEN pmm.filled_status = 'Filled' THEN 1 END) as total_filled_posts,
			COUNT(CASE WHEN pmm.filled_status = 'Vacant' OR pmm.filled_status IS NULL THEN 1 END) as total_vacant_posts
		FROM pmdm.kafka_office_master kom
		LEFT JOIN pmdm.kafka_office_hierarchy koh ON kom.office_id = koh.office_id
		LEFT JOIN office_with_subordinates ows ON kom.office_id = ows.direct_office_id
		LEFT JOIN pmdm.post_management_master pmm ON (
			pmm.office_id = ows.subordinate_office_id AND pmm.status = 'Active' AND pmm.cadre_name = $2
		)
		WHERE kom.reporting_office_id = $1 %s
		GROUP BY kom.office_id, kom.office_name, kom.office_type_code, kom.pincode, 
				kom.reporting_office_id, koh.division_name, koh.subdivision_name, koh.circle_name
		ORDER BY kom.office_name
		LIMIT $%d OFFSET $%d
	`, searchClause, len(params)-1, len(params))

	rows, err := r.db.Query(gctx, baseQuery, params...)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()

	for rows.Next() {
		var summary domain.HierarchySummary
		if err := rows.Scan(
			&summary.OfficeID,
			&summary.OfficeName,
			&summary.OfficeTypeCode,
			&summary.Pincode,
			&summary.ReportingOfficeID,
			&summary.DivisionName,
			&summary.SubdivisionName,
			&summary.CircleName,
			&summary.TotalPosts,
			&summary.TotalFilledPosts,
			&summary.TotalVacantPosts,
		); err != nil {
			return nil, 0, err
		}
		summaries = append(summaries, summary)
	}

	countQuery := fmt.Sprintf(`
		SELECT COUNT(DISTINCT kom.office_id) as total_count
		FROM pmdm.kafka_office_master kom
		LEFT JOIN pmdm.kafka_office_hierarchy koh2 ON kom.office_id = koh2.office_id
		WHERE kom.reporting_office_id = $1 %s
	`, searchClause)

	row := r.db.QueryRow(gctx, countQuery, countParams...)
	err = row.Scan(&totalCount)
	if err != nil {
		return nil, 0, err
	}

	return summaries, totalCount, nil
}

func (r *PostManagementRepository) GetHierarchyDetailList(ctx *gin.Context, cadreName string, parentOfficeID int64, search string) ([]domain.HierarchyDetail, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	params := []interface{}{parentOfficeID, cadreName}
	paramIndex := 3

	queryBuilder := strings.Builder{}
	queryBuilder.WriteString(`
		WITH RECURSIVE office_tree AS (
			SELECT office_id, office_id as root_office_id, 0 as level
			FROM pmdm.kafka_office_master 
			WHERE office_id = $1
			UNION ALL
			SELECT kom.office_id, ot.root_office_id, ot.level + 1
			FROM pmdm.kafka_office_master kom
			INNER JOIN office_tree ot ON kom.reporting_office_id = ot.office_id
			WHERE ot.level < 10
		),
		direct_reports AS (
			SELECT DISTINCT office_id FROM pmdm.kafka_office_master WHERE reporting_office_id = $1
		),
		office_with_subordinates AS (
			SELECT dr.office_id as direct_office_id, ot.office_id as subordinate_office_id
			FROM direct_reports dr
			LEFT JOIN office_tree ot ON (ot.root_office_id = dr.office_id OR ot.office_id = dr.office_id)
		)
		SELECT 
			kom.office_name as parent_office_name,
			kom.office_id as parent_office_id,
			pmm.postmanagement_id,
			pmm.post_id,
			pmm.post_name,
			pmm.designation,
			pmm.filled_status,
			pmm.post_status,
			pmm.office_name as post_office_name,
			pmm.office_id as post_office_id
		FROM pmdm.kafka_office_master kom
		INNER JOIN office_with_subordinates ows ON kom.office_id = ows.direct_office_id
		INNER JOIN pmdm.post_management_master pmm ON pmm.office_id = ows.subordinate_office_id
		WHERE kom.reporting_office_id = $1
		AND pmm.status = 'Active' 
		AND pmm.cadre_name = $2
	`)

	if search != "" {
		queryBuilder.WriteString(fmt.Sprintf(`
			AND (
				LOWER(kom.office_name) LIKE LOWER($%d) OR
				LOWER(pmm.post_name) LIKE LOWER($%d) OR
				LOWER(pmm.designation) LIKE LOWER($%d)
			)
		`, paramIndex, paramIndex, paramIndex))
		params = append(params, "%"+strings.ToLower(search)+"%")
	}

	queryBuilder.WriteString(" ORDER BY kom.office_name, pmm.post_name")

	rows, err := r.db.Query(gctx, queryBuilder.String(), params...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var details []domain.HierarchyDetail
	for rows.Next() {
		var d domain.HierarchyDetail
		if err := rows.Scan(
			&d.ParentOfficeName,
			&d.ParentOfficeID,
			&d.PostManagementID,
			&d.PostID,
			&d.PostName,
			&d.Designation,
			&d.FilledStatus,
			&d.PostStatus,
			&d.PostOfficeName,
			&d.PostOfficeID,
		); err != nil {
			return nil, err
		}
		details = append(details, d)
	}

	return details, nil
}

func (r *PostManagementRepository) GetHierarchyInfo(ctx *gin.Context, parentOfficeID int64) (domain.HierarchyInfodata, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	const query = `
		SELECT office_name 
		FROM pmdm.kafka_office_master 
		WHERE office_id = $1
	`

	row := r.db.QueryRow(gctx, query, parentOfficeID)

	var info domain.HierarchyInfodata
	info.ParentOfficeID = parentOfficeID

	err := row.Scan(&info.ParentOfficeName)
	if err != nil {
		return domain.HierarchyInfodata{
			ParentOfficeID:   parentOfficeID,
			ParentOfficeName: "Directorate",
		}, err
	}

	return info, nil

}

func (r *PostManagementRepository) GetOffices(ctx *gin.Context, divisionOfficeID int64, cadreName string, search *string, meta port.MetaDataRequest) ([]domain.OfficeData, int, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	params := []interface{}{divisionOfficeID, cadreName}
	queryIndex := 3
	whereClause := ""

	if search != nil {
		params = append(params, "%"+*search+"%")
		whereClause = fmt.Sprintf(`
			AND (
				LOWER(kom.office_name) LIKE LOWER($%d) OR
				LOWER(kom.office_type_code) LIKE LOWER($%d) OR
				LOWER(koh.division_name) LIKE LOWER($%d) OR
          		LOWER(koh.subdivision_name) LIKE LOWER($%d) OR
          		LOWER(pmm.group_name) LIKE LOWER($%d) OR
          		LOWER(pmm.cadre_name) LIKE LOWER($%d) OR
				kom.pincode::text LIKE $%d
			)
		`, queryIndex, queryIndex, queryIndex, queryIndex, queryIndex, queryIndex, queryIndex)
		queryIndex++
	}

	mainQuery := fmt.Sprintf(`
		SELECT 
			kom.office_id, kom.office_name, kom.office_type_code, kom.pincode, kom.email_id, kom.contact_number,
			koh.division_name, koh.division_office_id, koh.subdivision_name,
			pmm.group_name, pmm.cadre_name,
			COUNT(*) as total_posts,
			COUNT(CASE WHEN pmm.filled_status = 'Filled' THEN 1 END) as total_filled_posts,
			COUNT(CASE WHEN pmm.filled_status = 'Vacant' OR pmm.filled_status IS NULL THEN 1 END) as total_vacant_posts
		FROM pmdm.post_management_master pmm
		JOIN pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id AND koh.division_office_id = $1
		JOIN pmdm.kafka_office_master kom ON koh.office_id = kom.office_id
		WHERE pmm.status = 'Active' AND pmm.cadre_name = $2
		%s
		GROUP BY kom.office_id, kom.office_name, kom.office_type_code, kom.pincode, kom.email_id, kom.contact_number,
				 koh.division_name, koh.division_office_id, koh.subdivision_name, pmm.group_name, pmm.cadre_name
		ORDER BY kom.office_name
		LIMIT $%d OFFSET $%d
	`, whereClause, queryIndex, queryIndex+1)

	params = append(params, meta.Limit, meta.Skip)

	rows, err := r.db.Query(gctx, mainQuery, params...)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()

	var results []domain.OfficeData
	for rows.Next() {
		var o domain.OfficeData
		if err := rows.Scan(
			&o.OfficeID, &o.OfficeName, &o.OfficeTypeCode, &o.Pincode, &o.EmailID, &o.ContactNumber,
			&o.DivisionName, &o.DivisionOfficeID, &o.SubdivisionName,
			&o.GroupName, &o.CadreName,
			&o.TotalPosts, &o.TotalFilledPosts, &o.TotalVacantPosts,
		); err != nil {
			return nil, 0, err
		}
		results = append(results, o)
	}

	// Count query
	countQuery := `
		SELECT COUNT(DISTINCT koh.office_id)
		FROM pmdm.post_management_master pmm
		JOIN pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id AND koh.division_office_id = $1
		JOIN pmdm.kafka_office_master kom ON koh.office_id = kom.office_id
		WHERE pmm.status = 'Active' AND pmm.cadre_name = $2
	` + whereClause

	countRow := r.db.QueryRow(ctx, countQuery, params[:queryIndex-1]...)
	var totalCount int
	if err := countRow.Scan(&totalCount); err != nil {
		return nil, 0, err
	}

	return results, totalCount, nil
}

func (r *PostManagementRepository) GetDivisionInfo(ctx *gin.Context, divisionOfficeID int64) (*domain.OfficeInfo, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	row := r.db.QueryRow(gctx, `
		SELECT division_name, region_name, region_office_id, circle_name, circle_office_id
		FROM pmdm.kafka_office_hierarchy
		WHERE division_office_id = $1
		LIMIT 1
	`, divisionOfficeID)

	var info domain.OfficeInfo
	err := row.Scan(&info.DivisionName, &info.RegionName, &info.RegionOfficeID, &info.CircleName, &info.CircleOfficeID)
	if err != nil {
		return nil, err
	}
	return &info, nil
}

func (r *PostManagementRepository) DetailOfficeReport(ctx *gin.Context, cadreName string, divisionOfficeID int64, officeID int64, search *string) ([]domain.OfficePostDetail, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	params := []interface{}{cadreName, divisionOfficeID, officeID}
	paramIndex := 4

	searchClause := ""
	if search != nil {
		params = append(params, "%"+*search+"%")
		searchClause = fmt.Sprintf(`
			AND (
				LOWER(pmm.post_name) LIKE LOWER($%d) OR
				LOWER(pmm.designation) LIKE LOWER($%d) OR
				LOWER(kom.office_name) LIKE LOWER($%d)
			)
		`, paramIndex, paramIndex, paramIndex)
	}

	query := fmt.Sprintf(`
		SELECT 
			pmm.postmanagement_id,
			pmm.post_id,
			pmm.post_name,
			pmm.designation,
			pmm.filled_status,
			pmm.post_status,
			pmm.pay_level,
			pmm.grade_pay,
			pmm.sanctioned_strength,
			pmm.permanent_status,
			pmm.allowances_attached,
			pmm.allowance_description,
			pmm.group_name,
			pmm.cadre_name,
			pmm.office_name AS post_office_name,
			pmm.office_id,
			kom.office_name,
			kom.office_type_code,
			kom.pincode,
			koh.division_name,
			koh.subdivision_name,
			koh.circle_name,
			koh.region_name
		FROM pmdm.post_management_master pmm
		LEFT JOIN pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id
		LEFT JOIN pmdm.kafka_office_master kom ON pmm.office_id = kom.office_id
		WHERE pmm.status = 'Active'
		  AND pmm.cadre_name = $1
		  AND koh.division_office_id = $2
		  AND pmm.office_id = $3
		%s
		ORDER BY pmm.post_name, pmm.designation
	`, searchClause)

	rows, err := r.db.Query(gctx, query, params...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []domain.OfficePostDetail
	for rows.Next() {
		var d domain.OfficePostDetail
		err := rows.Scan(
			&d.PostManagementID,
			&d.PostID,
			&d.PostName,
			&d.Designation,
			&d.FilledStatus,
			&d.PostStatus,
			&d.PayLevel,
			&d.GradePay,
			&d.SanctionedStrength,
			&d.PermanentStatus,
			&d.AllowancesAttached,
			&d.AllowanceDescription,
			&d.GroupName,
			&d.CadreName,
			&d.PostOfficeName,
			&d.OfficeID,
			&d.OfficeName,
			&d.OfficeTypeCode,
			&d.Pincode,
			&d.DivisionName,
			&d.SubdivisionName,
			&d.CircleName,
			&d.RegionName,
		)
		if err != nil {
			return nil, err
		}
		results = append(results, d)
	}

	return results, nil
}

func (r *PostManagementRepository) FetchPosts(ctx *gin.Context, params domain.PostReport) ([]domain.PostDetail, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	var (
		whereClauses = []string{"pmm.status = 'Active'", "pmm.cadre_id = $1"}
		queryParams  = []interface{}{params.CadreID}
		paramIndex   = 2
	)

	// Dynamic filter by hierarchy
	if params.OfficeID != nil {
		whereClauses = append(whereClauses, fmt.Sprintf("pmm.office_id = $%d", paramIndex))
		queryParams = append(queryParams, params.OfficeID)
		paramIndex++
	} else if params.DivisionOfficeID != nil {
		whereClauses = append(whereClauses, fmt.Sprintf("koh.division_office_id = $%d", paramIndex))
		queryParams = append(queryParams, *params.DivisionOfficeID)
		paramIndex++
	} else if params.RegionOfficeID != nil {
		whereClauses = append(whereClauses, fmt.Sprintf("koh.region_office_id = $%d", paramIndex))
		queryParams = append(queryParams, *params.RegionOfficeID)
		paramIndex++
	} else if params.CircleOfficeID != nil {
		whereClauses = append(whereClauses, fmt.Sprintf("koh.circle_office_id = $%d", paramIndex))
		queryParams = append(queryParams, *params.CircleOfficeID)
		paramIndex++
	}

	query := `
		SELECT 
			pmm.postmanagement_id, pmm.post_name, pmm.designation, pmm.filled_status, pmm.post_status,
			pmm.pay_level, pmm.grade_pay, pmm.sanctioned_strength, pmm.permanent_status,
			pmm.allowances_attached, pmm.allowance_description, pmm.group_name, pmm.cadre_name,
			pmm.office_name, pmm.office_id, kom.office_name, kom.office_type_code, kom.pincode,
			koh.division_name, koh.subdivision_name, koh.circle_name, koh.region_name
		FROM pmdm.post_management_master pmm
		LEFT JOIN pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id
		LEFT JOIN pmdm.kafka_office_master kom ON pmm.office_id = kom.office_id
		WHERE ` + strings.Join(whereClauses, " AND ")

	if params.Search != "" {
		searchParam := fmt.Sprintf("LOWER(pmm.post_name) LIKE LOWER($%d) OR LOWER(pmm.designation) LIKE LOWER($%d) OR LOWER(pmm.filled_status) LIKE LOWER($%d) OR LOWER(kom.office_name) LIKE LOWER($%d)", paramIndex, paramIndex, paramIndex, paramIndex)
		query += " AND (" + searchParam + ")"
		queryParams = append(queryParams, "%"+params.Search+"%")
	}

	query += " ORDER BY kom.office_name, pmm.post_name, pmm.designation"

	rows, err := r.db.Query(gctx, query, queryParams...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var posts []domain.PostDetail
	for rows.Next() {
		var post domain.PostDetail
		if err := rows.Scan(
			&post.PostManagementID, &post.PostName, &post.Designation, &post.FilledStatus, &post.PostStatus,
			&post.PayLevel, &post.GradePay, &post.SanctionedStrength, &post.PermanentStatus,
			&post.AllowancesAttached, &post.AllowanceDescription, &post.GroupName, &post.CadreName,
			&post.PostOfficeName, &post.OfficeID, &post.OfficeName, &post.OfficeTypeCode, &post.Pincode,
			&post.DivisionName, &post.SubdivisionName, &post.CircleName, &post.RegionName,
		); err == nil {
			posts = append(posts, post)
		}
	}
	return posts, nil
}

func (r *PostManagementRepository) GetContextInfo(ctx *gin.Context, officeID int64) (domain.ContextInfo, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	var info domain.ContextInfo
	query := `
		SELECT kom.office_name, kom.office_type_code, koh.division_name, koh.region_name, koh.circle_name
		FROM pmdm.kafka_office_master kom
		LEFT JOIN pmdm.kafka_office_hierarchy koh ON kom.office_id = koh.office_id
		WHERE kom.office_id = $1
	`
	err := r.db.QueryRow(gctx, query, officeID).Scan(
		&info.OfficeName, &info.OfficeTypeCode, &info.DivisionName, &info.RegionName, &info.CircleName,
	)
	return info, err
}

func (r *PostManagementRepository) CountStatus(posts []domain.PostDetail, status string) int {
	count := 0
	for _, p := range posts {
		if strings.EqualFold(p.FilledStatus, status) || (status == "Vacant" && p.FilledStatus == "") {
			count++
		}
	}
	return count
}

func (r *PostManagementRepository) FetchRegionSummary(ctx *gin.Context, req domain.RegionRequest, meta port.MetaDataRequest) ([]domain.RegionSummary, int, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	var summaries []domain.RegionSummary
	var totalCount int
	var queryParams []interface{}
	paramIndex := 3

	baseQuery := `
		SELECT 
			koh.region_name,
			koh.region_office_id,
			koh.circle_name,
			koh.circle_office_id,
			pmm.group_name,
			pmm.cadre_name,
			COUNT(*) as total_posts,
			COUNT(CASE WHEN pmm.filled_status = 'Filled' THEN 1 END) as total_filled_posts,
			COUNT(CASE WHEN pmm.filled_status = 'Vacant' OR pmm.filled_status IS NULL THEN 1 END) as total_vacant_posts
		FROM pmdm.post_management_master pmm
		JOIN pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id 
			AND koh.circle_office_id = $1 
			AND koh.region_office_id IS NOT NULL
			AND koh.region_office_id <> 0
		WHERE pmm.status = 'Active' AND pmm.cadre_name = $2
	`
	countQuery := `
		SELECT COUNT(DISTINCT koh.region_office_id) FROM pmdm.post_management_master pmm
		JOIN pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id 
			AND koh.circle_office_id = $1 
			AND koh.region_office_id IS NOT NULL
			AND koh.region_office_id <> 0
		WHERE pmm.status = 'Active' AND pmm.cadre_name = $2
	`

	queryParams = append(queryParams, req.CircleOfficeID, req.CadreName)

	if req.Search != "" {
		search := "%" + strings.ToLower(req.Search) + "%"
		baseQuery += fmt.Sprintf(`
			AND (
				LOWER(koh.region_name) LIKE $%d OR 
				LOWER(koh.circle_name) LIKE $%d OR 
				LOWER(pmm.group_name) LIKE $%d OR 
				LOWER(pmm.cadre_name) LIKE $%d
			)
		`, paramIndex, paramIndex, paramIndex, paramIndex)
		countQuery += fmt.Sprintf(`
			AND (
				LOWER(koh.region_name) LIKE $%d OR 
				LOWER(koh.circle_name) LIKE $%d OR 
				LOWER(pmm.group_name) LIKE $%d OR 
				LOWER(pmm.cadre_name) LIKE $%d
			)
		`, paramIndex, paramIndex, paramIndex, paramIndex)
		queryParams = append(queryParams, search)
		paramIndex++
	}

	offset := (meta.Skip - 1) * meta.Limit
	baseQuery += fmt.Sprintf(`
		GROUP BY koh.region_name, koh.region_office_id, koh.circle_name, koh.circle_office_id,
		         pmm.group_name, pmm.cadre_name
		ORDER BY koh.region_name, pmm.group_name, pmm.cadre_name
		LIMIT $%d OFFSET $%d
	`, paramIndex, paramIndex+1)

	queryParams = append(queryParams, meta.Limit, offset)

	rows, err := r.db.Query(gctx, baseQuery, queryParams...)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()

	for rows.Next() {
		var row domain.RegionSummary
		if err := rows.Scan(
			&row.RegionName, &row.RegionOfficeID,
			&row.CircleName, &row.CircleOfficeID,
			&row.GroupName, &row.CadreName,
			&row.TotalPosts, &row.TotalFilledPosts, &row.TotalVacantPosts,
		); err != nil {
			return nil, 0, err
		}
		summaries = append(summaries, row)
	}

	// Fetch count
	err = r.db.QueryRow(gctx, countQuery, queryParams[:paramIndex-1]...).Scan(&totalCount)
	if err != nil {
		return nil, 0, err
	}

	return summaries, totalCount, nil
}

func (r *PostManagementRepository) FetchRegionDetails(ctx *gin.Context, req domain.RegionRequest) ([]domain.RegionDetail, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	var details []domain.RegionDetail
	var queryParams []interface{}
	paramIndex := 4

	query := `
		SELECT 
			pmm.postmanagement_id, pmm.post_name, pmm.designation, pmm.filled_status, pmm.post_status,
			pmm.pay_level, pmm.grade_pay, pmm.sanctioned_strength, pmm.permanent_status,
			pmm.allowances_attached, pmm.allowance_description, pmm.group_name, pmm.cadre_name,
			pmm.office_name as post_office_name, pmm.office_id, kom.office_name, kom.office_type_code,
			kom.pincode, koh.division_name, koh.subdivision_name, koh.circle_name, koh.region_name
		FROM pmdm.post_management_master pmm
		LEFT JOIN pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id
		LEFT JOIN pmdm.kafka_office_master kom ON pmm.office_id = kom.office_id
		WHERE pmm.status = 'Active' AND pmm.cadre_name = $1 AND koh.circle_office_id = $2 AND koh.region_office_id = $3
	`
	queryParams = append(queryParams, req.CadreName, req.CircleOfficeID, req.RegionOfficeID)

	if req.Search != "" {
		search := "%" + strings.ToLower(req.Search) + "%"
		query += fmt.Sprintf(`
			AND (
				LOWER(pmm.post_name) LIKE $%d OR
				LOWER(pmm.designation) LIKE $%d OR
				LOWER(kom.office_name) LIKE $%d
			)
		`, paramIndex, paramIndex, paramIndex)
		queryParams = append(queryParams, search)
	}

	query += ` ORDER BY kom.office_name, pmm.post_name, pmm.designation`

	rows, err := r.db.Query(gctx, query, queryParams...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var row domain.RegionDetail
		if err := rows.Scan(
			&row.PostManagementID, &row.PostName, &row.Designation, &row.FilledStatus, &row.PostStatus,
			&row.PayLevel, &row.GradePay, &row.SanctionedStrength, &row.PermanentStatus,
			&row.AllowancesAttached, &row.AllowanceDesc, &row.GroupName, &row.CadreName,
			&row.PostOfficeName, &row.OfficeID, &row.OfficeName, &row.OfficeTypeCode,
			&row.Pincode, &row.DivisionName, &row.SubdivisionName, &row.CircleName, &row.RegionName,
		); err != nil {
			return nil, err
		}
		details = append(details, row)
	}
	return details, nil
}

func (r *PostManagementRepository) GetCircleInfo(ctx context.Context, circleOfficeID int64) (*domain.CircleInfo, error) {
	query := `
		SELECT DISTINCT circle_name, circle_code 
		FROM pmdm.kafka_office_hierarchy 
		WHERE circle_office_id = $1
		LIMIT 1
	`
	var info domain.CircleInfo
	err := r.db.QueryRow(ctx, query, circleOfficeID).Scan(&info.CircleName, &info.CircleCode)
	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, nil // not found, return nil
		}
		return nil, err
	}
	return &info, nil
}

func (r *PostManagementRepository) CalculateSummary(summaries []domain.RegionSummary) map[string]int {
	var totalPosts, totalFilled, totalVacant int

	for _, s := range summaries {
		totalPosts += s.TotalPosts
		totalFilled += s.TotalFilledPosts
		totalVacant += s.TotalVacantPosts
	}

	return map[string]int{
		"totalPosts":  totalPosts,
		"totalFilled": totalFilled,
		"totalVacant": totalVacant,
	}
}

func (rmr *PostManagementRepository) GetListCadreWiseOfficeWiseReports(
	ctx *gin.Context, divisionOfficeID int, cadreId int,
) ([]domain.ListCadreWiseReport, error) {

	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Minute)
	defer cancel()

	query := dblib.Psql.Select(
		"pmm.post_id",
		"pmm.post_name",
		"pmm.office_id",
		"pmm.office_name",
		"pmm.cadre_id",
		"pmm.cadre_name",
		"pmm.designation_id",
		"pmm.designation",
		"'Vacant' as filled_status",
	).
		Distinct().
		From("pmdm.post_management_master pmm").
		Join("pmdm.kafka_office_hierarchy koh ON pmm.office_id = koh.office_id").
		LeftJoin("pmdm.kafka_employee_master kem ON pmm.post_id = kem.post_id").
		Where(sq.And{
			sq.Eq{"koh.division_office_id": divisionOfficeID},
			sq.Eq{"pmm.cadre_id": cadreId},
			//sq.Eq{"pmm.filled_status": "Vacant"},
			sq.Or{
				sq.Eq{"kem.employee_id": nil},
				sq.NotEq{"kem.employment_status": "Active"},
			},
			sq.Eq{"pmm.status": "Active"},
		}).
		OrderBy("pmm.post_id ASC")

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.ListCadreWiseReport])
}

func (rmr *PostManagementRepository) GetCadreWiseOfficeWiseReports(ctx *gin.Context, officeID int, cadreID int) ([]domain.CadreWiseOfficeWiseReport, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := `
        WITH office_scope AS (
    SELECT
        pm.*,
        ohm.circle_office_id,
        ohm.region_office_id,
        ohm.division_office_id,
        ohm.sub_division_office_id,
        om.office_name AS office_name_in_scope,
        CAST($1 AS BIGINT) AS group_office_id,
        -- Level just below input
        CASE 
            WHEN CAST($1 AS BIGINT) = ohm.circle_office_id THEN ohm.region_office_id
            WHEN CAST($1 AS BIGINT) = ohm.region_office_id THEN ohm.division_office_id
            WHEN CAST($1 AS BIGINT) = ohm.division_office_id THEN ohm.sub_division_office_id
            WHEN CAST($1 AS BIGINT) = ohm.sub_division_office_id THEN pm.office_id
            ELSE NULL
        END AS effective_office_id
    FROM pmdm.post_management_master pm
    LEFT JOIN pmdm.kafka_office_hierarchy ohm ON pm.office_id = ohm.office_id
    JOIN pmdm.kafka_office_master om ON pm.office_id = om.office_id
    WHERE pm.status = 'Active'
      AND (
          (CAST($1 AS BIGINT) = ohm.circle_office_id AND ohm.region_office_id IS NOT NULL)
          OR (CAST($1 AS BIGINT) = ohm.region_office_id AND ohm.division_office_id IS NOT NULL)
          OR (CAST($1 AS BIGINT) = ohm.division_office_id AND ohm.sub_division_office_id IS NOT NULL)
          OR (CAST($1 AS BIGINT) = ohm.sub_division_office_id AND pm.office_id IS NOT NULL)
      )
      AND pm.cadre_id = $2
)
SELECT
    os.group_office_id,
    gom.office_name AS group_office_name,
    os.effective_office_id AS office_id,
    em.office_name AS office_name,
    os.cadre_id,
    os.cadre_name,
    COALESCE(COUNT(DISTINCT os.post_id), 0) AS total_posts,
    COALESCE(SUM(
        CASE 
            WHEN kem.employee_id IS NOT NULL AND kem.employment_status = 'Active' THEN 1
            ELSE 0 
        END
    ), 0) AS total_filled_posts,
    COALESCE(SUM(
        CASE 
            WHEN kem.employee_id IS NULL OR kem.employment_status != 'Active' THEN 1
            ELSE 0 
        END
    ), 0) AS total_vacant_posts
FROM office_scope os
LEFT JOIN pmdm.kafka_employee_master kem ON os.post_id = kem.post_id
-- group office name
JOIN pmdm.kafka_office_master gom ON os.group_office_id = gom.office_id
-- office name of one-level-below office
JOIN pmdm.kafka_office_master em ON os.effective_office_id = em.office_id
GROUP BY os.group_office_id, gom.office_name, os.effective_office_id, em.office_name, os.cadre_id, os.cadre_name
ORDER BY os.group_office_id, os.cadre_name;

    `
	rows, err := rmr.db.Query(gctx, query, officeID, cadreID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []domain.CadreWiseOfficeWiseReport
	for rows.Next() {
		var row domain.CadreWiseOfficeWiseReport
		if err := rows.Scan(
			&row.GroupOfficeID,
			&row.GroupOfficeName,
			&row.OfficeID,
			&row.OfficeName,
			&row.CadreID,
			&row.CadreName,
			&row.TotalPosts,
			&row.TotalFilledPosts,
			&row.TotalVacantPosts,
		); err != nil {
			return nil, err
		}
		results = append(results, row)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return results, nil
}
func (pmr *PostManagementRepository) FetchAllActivePostByOfficeCadre(gctx *gin.Context, officeID int, cadreId string) ([]domain.PostManagementMasterNew, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.
		Select(
			"COALESCE(pm.office_id, 0) AS office_id",
			"COALESCE(pm.cadre_id, 0) AS cadre_id",
			"COALESCE(pm.cadre_name, '') AS cadre_name",
			"COALESCE(pm.group_id, 0) AS group_id",
			"COALESCE(pm.group_name, '') AS group_name",
			"COALESCE(pm.employee_group, '') AS employee_group",
			"COALESCE(pm.post_id, 0) AS post_id",
			"COALESCE(pm.post_name, '') AS post_name",
			"COALESCE(pm.designation_id, 0) AS designation_id",
			"COALESCE(pm.designation, '') AS designation",
			`CASE 
			WHEN km.employee_id IS NULL OR km.employment_status != 'Active' 
			THEN 'Vacant'
			ELSE 'Filled'
		END AS filled_status`,
			"COALESCE(kom.office_name, '') AS office_name",
			"COALESCE(km.post_status, '') AS post_status",
			"COALESCE(km.employee_id, 0) AS employee_id",
			"COALESCE(CONCAT(km.employee_first_name, ' ', km.employee_middle_name, ' ', km.employee_last_name), '') AS employee_name",
			"COALESCE(pm.is_head_of_the_office, false) AS is_head_of_the_office",
			"COALESCE(pmd.employee_post_id, 0) AS employee_post_id",
			"COALESCE(pmd.leave_sanc_authority_1, 0) AS leave_sanc_authority_1",
			"COALESCE(pmd.leave_sanc_authority_2, 0) AS leave_sanc_authority_2",
			"COALESCE(pmd.pay_approve_authority1, 0) AS pay_approve_authority1",
			"COALESCE(pmd.appointing_authority, 0) AS appointing_authority",
			"COALESCE(pmd.disciplinary_authority, 0) AS disciplinary_authority",
			"COALESCE(pmd.ddo_authority, 0) AS ddo_authority",
			"COALESCE(pmd.employee_office_id, 0) AS employee_office_id",
			"COALESCE(pmd.vigilence_maker_authority, 0) AS vigilence_maker_authority",
			"COALESCE(pm.permanent_status, false) AS permanent_status",
			"COALESCE(pm.allowances_attached, false) AS allowances_attached",
			"COALESCE(pm.allowance_description, '') AS allowance_description",
			"pm.status",
			"pm.created_by",
			"COALESCE(pm.created_date, '1970-01-01'::timestamp) AS created_date",
			"pm.approved_by",
			"COALESCE(pm.approved_date, '1970-01-01'::timestamp) AS approved_date",
			"pm.updated_by",
			"COALESCE(pm.updated_date, '1970-01-01'::timestamp) AS updated_date",
			"COALESCE(pm.valid_from, '1970-01-01'::timestamp) AS valid_from",
			"COALESCE(pm.valid_to, '1970-01-01'::timestamp) AS valid_to",
			"pm.order_casemark",
			"COALESCE(pm.order_date, '1970-01-01'::timestamp) AS order_date",
			"COALESCE(pm.upload_order_doc_name, '') AS upload_order_doc_name",
			"COALESCE(pm.establishment_register_id, 0) AS establishment_register_id",
			"COALESCE(pm.establishment_register_name, '') AS establishment_register_name",
			"COALESCE(pm.remarks, '') AS remarks",
		).
		From("pmdm.post_management_master pm").
		LeftJoin("pmdm.kafka_employee_master km ON km.post_id = pm.post_id AND km.employment_status = 'Active'").
		LeftJoin("pmdm.post_mapping_detail pmd ON pmd.employee_post_id = pm.post_id").
		LeftJoin("kafka_office_master kom ON pm.office_id = kom.office_id").
		Where(sq.And{
			sq.Eq{"pm.office_id": officeID},
			sq.Eq{"pm.status": "Active"},
			sq.Eq{"pm.cadre_id": cadreId},
		})

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMasterNew])
}

func (pmr *PostManagementRepository) DeletePostsbyOfficeIDRepo(gctx *gin.Context, officeID int, userID string) (string, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	// Prepare update query
	query := dblib.Psql.Update("pmdm.post_management_master").
		Set("status", "Inactive").
		Set("Filled_status", "Vacant").
		Set("post_status", "Inactive").
		Set("valid_to", time.Now()).
		Set("updated_by", userID).
		Set("updated_date", time.Now()).
		Where(sq.Eq{"office_id": officeID})

	query1 := dblib.Psql.Update("pmdm.post_management_master_maker").
		Set("status", "Inactive").
		Set("Filled_status", "Vacant").
		Set("post_status", "Inactive").
		Set("valid_to", time.Now()).
		Set("updated_by", userID).
		Set("updated_date", time.Now()).
		Where(sq.Eq{"office_id": officeID})

	// Execute update
	result, err := dblib.Update(ctx, pmr.db, query)
	if err != nil {
		log.Error(gctx, "Error deleting posts for office ID:", officeID, "Error:", err)
		return "", err
	}

	// Execute update2
	result1, err1 := dblib.Update(ctx, pmr.db, query1)
	if err != nil {
		log.Error(gctx, "Error deleting posts for office ID:", officeID, "Error:", err1)
		return "", err1
	}
	// Check affected rows
	rowsAffected := result.RowsAffected()
	if rowsAffected == 0 {
		log.Warn(gctx, "No rows were affected by the update query for office ID:", officeID)
		return "", errors.New("no posts found to delete for the given office ID")
	}

	rowsAffected1 := result1.RowsAffected()
	if rowsAffected1 == 0 {
		log.Warn(gctx, "No rows were affected by the update query for office ID:", officeID)
		return "", errors.New("no posts found to delete for the given office ID")
	}
	log.Info(gctx, "Posts deleted successfully for office ID:", officeID, "Rows affected:", rowsAffected)
	return "Posts deleted successfully", nil
}

func (pmr *PostManagementRepository) CheckCadreExistsRepo(gctx *gin.Context, postID int) (bool, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	// valid cadre_id -> cadre_name mapping
	validCadres := map[int]string{
		103: "IPoS_HAG",
		102: "IPoS_HAG+",
		106: "IPoS_JAG",
		105: "IPoS_JAG_NFSG",
		107: "IPoS_JTS",
		104: "IPoS_SAG",
		2:   "IPoS_STS",
		10:  "Postal Service GrP B",
	}

	// holder for db values
	var cadreID int
	var cadreName string

	query := dblib.Psql.
		Select("cadre_id", "cadre_name").
		From("pmdm.post_management_master").
		Where(sq.Eq{"post_id": postID}).
		Limit(1)

	sqlStr, args, err := query.ToSql()
	if err != nil {
		return false, err
	}

	err = pmr.db.QueryRow(ctx, sqlStr, args...).Scan(&cadreID, &cadreName)
	if err != nil {
		if err == pgx.ErrNoRows {
			return false, nil
		}
		return false, err
	}

	// check validity
	if name, ok := validCadres[cadreID]; ok && name == cadreName {
		return true, nil
	}

	return false, nil
}

func (r *PostManagementRepository) GetSanctionedStrengthByOfficeIDRepo(
	ctx *gin.Context,
	officeId int64,
	metadata port.MetaDataRequest,
) ([]domain.SanctionedStrengthDetails, error) {

	query := dblib.Psql.
		Select(
			"COALESCE(pmm.group_id, 0) AS group_id",
			"COALESCE(pmm.cadre_id, 0) AS cadre_id",
			"COALESCE(cm.cadre_name, '') AS cadre_name",
			"COALESCE(pmm.office_id, 0) AS office_id",
			"COALESCE(kom.office_name, '') AS office_name",
			"COALESCE(COUNT(pmm.post_id), 0) AS sanctioned_strength",
		).
		From("pmdm.post_management_master pmm").
		Join("pmdm.kafka_office_master kom ON kom.office_id = pmm.office_id").
		Join("pmdm.cadre_master cm ON cm.cadre_id = pmm.cadre_id").
		Where(sq.Eq{
			"pmm.status": []string{"Active", "Abeyance", "Surplus"},
		}).
		Where(sq.Eq{"pmm.office_id": officeId}).
		GroupBy(
			"pmm.group_id",
			"pmm.cadre_id",
			"cm.cadre_name",
			"pmm.office_id",
			"kom.office_name",
		).
		Offset(uint64(metadata.Skip * metadata.Limit)).
		Limit(uint64(metadata.Limit))

	sqlStr, args, err := query.ToSql()
	if err != nil {
		return nil, fmt.Errorf("failed to build SQL: %w", err)
	}

	rows, err := r.db.Query(ctx, sqlStr, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to execute query: %w", err)
	}
	defer rows.Close()

	var result []domain.SanctionedStrengthDetails

	for rows.Next() {
		var item domain.SanctionedStrengthDetails

		err = rows.Scan(
			&item.GroupID,
			&item.CadreID,
			&item.CadreName,
			&item.OfficeID,
			&item.OfficeName,
			&item.SanctionedStrength,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}

		result = append(result, item)
	}

	return result, nil
}

func (ap *PostManagementRepository) UpdatePostNamebyPostIDRepo(
	gctx *gin.Context,
	postId int,
	postName string,
	remarks string,
	updatedBy string,
) (string, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), ap.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := `
		UPDATE pmdm.post_management_master
		SET 
			post_name = $2,
			remarks = $3,
			updated_date = NOW(),
			updated_by = $4
		WHERE post_id = $1
	`

	_, err := ap.db.Exec(ctx, query,
		postId,
		postName,
		remarks,
		updatedBy,
	)

	if err != nil {
		log.Error(gctx, "Failed to update post name", "error", err)
		return "", fmt.Errorf("failed to update post (post_id=%d): %w", postId, err)
	}

	return "Post updated successfully", nil
}

func (r *PostManagementRepository) GetPostDetailsbyPostID1(ctx *gin.Context, postID int) (domain.Postdetails, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := `SELECT 
	postmanagement_id,
    COALESCE(post_id, 0)                                      AS post_id,
    COALESCE(office_id, 0)                                    AS office_id,
	COALESCE(office_name, '')                                  AS office_name,
    COALESCE(post_name, '')                                   AS post_name,
    COALESCE(group_id, 0)                                     AS group_id,
    COALESCE(cadre_name, '')                                  AS cadre_name,
    COALESCE(filled_status, '')                               AS filled_status,
    COALESCE(status, '')                                      AS status,
    COALESCE(remarks, '')                                     AS remarks,
    COALESCE(valid_from, '1970-01-01'::timestamp)             AS valid_from,
    COALESCE(valid_to, '1970-01-01'::timestamp)               AS valid_to,
    COALESCE(order_casemark, '')                              AS order_casemark,
    COALESCE(order_date, '1970-01-01'::timestamp)             AS order_date,
    COALESCE(designation, '')                                 AS designation,
    COALESCE(pay_level, 0)                                    AS pay_level,
    COALESCE(grade_pay, 0)                                    AS grade_pay,
    COALESCE(permanent_status, false)                         AS permanent_status,
    COALESCE(group_name, '')                                  AS group_name,
    COALESCE(cadre_id, 0)                                     AS cadre_id,
    COALESCE(designation_id, 0)                               AS designation_id,
    COALESCE(is_head_of_the_office, false)                    AS is_head_of_the_office
FROM pmdm.post_management_master
WHERE post_id = $1;
;`

	var d domain.Postdetails
	err := r.db.QueryRow(gctx, query, postID).Scan(
		&d.PostManagementID,
		&d.PostID,
		&d.OfficeID,
		&d.OfficeName,
		&d.PostName,
		&d.GroupID,
		&d.CadreName,
		&d.FilledStatus,
		&d.Status,
		&d.Remarks,
		&d.ValidFrom,
		&d.ValidTo,
		&d.OrderCasemark,
		&d.OrderDate,
		&d.Designation,
		&d.PayLevel,
		&d.GradePay,
		&d.PermanentStatus,
		&d.GroupName,
		&d.CadreID,
		&d.DesignationID,
		&d.IsHeadOfTheOffice,
	)
	if err != nil {
		return domain.Postdetails{}, err
	}
	return d, nil
}

func (r *PostManagementRepository) GetPostSummaryRepo(
	gctx *gin.Context,
	cadreName string,
	circleOfficeID int64,
	includeList bool,
	meta port.MetaDataRequest,
) (*domain.PostSummaryResponse, error) {

	ctx, cancel := context.WithTimeout(
		gctx.Request.Context(),
		r.cfg.GetDuration(DBtimeout),
	)
	defer cancel()

	if includeList {
		list, total, err := r.GetCadreSummaryWithList(ctx, cadreName, circleOfficeID, meta)
		if err != nil {
			return nil, err
		}

		return &domain.PostSummaryResponse{
			List:  list,
			Total: total,
		}, nil
	}

	summary, total, err := r.GetCadreSummary1(ctx, cadreName, meta)
	if err != nil {
		return nil, err
	}

	return &domain.PostSummaryResponse{
		Summary: summary,
		Total:   total,
	}, nil
}

func (r *PostManagementRepository) GetCadreSummary1(
	ctx context.Context,
	cadreName string,
	meta port.MetaDataRequest,
) ([]domain.PostDetail1, int, error) {

	query := `
	select
	koh.circle_name,
	koh.circle_office_id,
	COALESCE(pmm.group_name, '') as group_name,
	pmm.cadre_name,
	COUNT(*) as total_posts,
	COUNT(case when pmm.filled_status = 'Filled' then 1 end) as total_filled_posts,
	COUNT(case when pmm.filled_status = 'Vacant' or pmm.filled_status is null then 1 end) as total_vacant_posts
from
	pmdm.post_management_master pmm
join pmdm.kafka_office_hierarchy koh on
	pmm.office_id = koh.office_id
where
	pmm.status = 'Active'
	and pmm.cadre_name = $1
	and koh.circle_office_id is not null
	and koh.circle_office_id <> 0
group by
	koh.circle_name,
	koh.circle_office_id,
	pmm.group_name,
	pmm.cadre_name
order by
	koh.circle_name,
	pmm.group_name,
	pmm.cadre_name
limit $2 offset $3
	`

	rows, err := r.db.Query(
		ctx,
		query,
		cadreName,
		meta.Limit,
		meta.Skip,
	)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()

	var result []domain.PostDetail1

	for rows.Next() {
		var cs domain.PostDetail1
		if err := rows.Scan(
			&cs.CircleName,
			&cs.CircleOfficeID,
			&cs.GroupName,
			&cs.CadreName,
			&cs.TotalPosts,
			&cs.TotalFilledPosts,
			&cs.TotalVacantPosts,
		); err != nil {
			return nil, 0, err
		}
		result = append(result, cs)
	}

	countQuery := `
	select
	COUNT(distinct koh.circle_office_id) as total_count
from
	pmdm.post_management_master pmm
join pmdm.kafka_office_hierarchy koh on
	pmm.office_id = koh.office_id
where
	pmm.status = 'Active'
	and pmm.cadre_name = $1
	and koh.circle_office_id is not null
	and koh.circle_office_id <> 0
	`

	var total int
	if err := r.db.QueryRow(ctx, countQuery, cadreName).Scan(&total); err != nil {
		return nil, 0, err
	}

	return result, total, nil
}

func (r *PostManagementRepository) GetCadreSummaryWithList(
	ctx context.Context,
	cadreName string,
	circleOfficeID int64,
	meta port.MetaDataRequest,
) ([]domain.PostSummaryDetail, int, error) {

	// Main query with COALESCE for nullable columns
	query := `
SELECT
	pmm.postmanagement_id,
	COALESCE(pmm.post_name, '')                    AS post_name,
	COALESCE(pmm.designation, '')                  AS designation,
	COALESCE(pmm.filled_status, '')                AS filled_status,
	COALESCE(pmm.post_status, '')                  AS post_status,
	COALESCE(pmm.pay_level, 0)                    AS pay_level,
	COALESCE(pmm.grade_pay, 0)                    AS grade_pay,
	COALESCE(pmm.sanctioned_strength, 0)          AS sanctioned_strength,
	COALESCE(pmm.permanent_status, false)         AS permanent_status,
	COALESCE(pmm.allowances_attached, false)      AS allowances_attached,
	COALESCE(pmm.allowance_description, '')       AS allowance_description,
	COALESCE(pmm.group_name, '')                   AS group_name,
	COALESCE(pmm.cadre_name, '')                   AS cadre_name,
	COALESCE(pmm.office_name, '')                  AS post_office_name,
	COALESCE(pmm.office_id, 0)                     AS office_id,
	COALESCE(kom.office_name, '')                  AS office_name,
	COALESCE(kom.office_type_code, '')             AS office_type_code,
	COALESCE(kom.pincode, 0)                       AS pincode,
	COALESCE(koh.division_name, '')                AS division_name,
	COALESCE(koh.subdivision_name, '')             AS subdivision_name,
	COALESCE(koh.circle_name, '')                   AS circle_name,
	COALESCE(koh.region_name, '')                   AS region_name
from
	pmdm.post_management_master pmm
left join pmdm.kafka_office_hierarchy koh on
	pmm.office_id = koh.office_id
left join pmdm.kafka_office_master kom on
	pmm.office_id = kom.office_id
where
	pmm.status = 'Active'
	and pmm.cadre_name = $1
	and koh.circle_office_id = $2
order by
	kom.office_name,
	pmm.post_name,
	pmm.designation
`

	rows, err := r.db.Query(ctx, query, cadreName, circleOfficeID)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()

	var result []domain.PostSummaryDetail

	for rows.Next() {
		var p domain.PostSummaryDetail

		if err := rows.Scan(
			&p.PostManagementID,
			&p.PostName,
			&p.Designation,
			&p.FilledStatus,
			&p.PostStatus,
			&p.PayLevel,
			&p.GradePay,
			&p.SanctionedStrength,
			&p.PermanentStatus,
			&p.AllowancesAttached,
			&p.AllowanceDescription,
			&p.GroupName,
			&p.CadreName,
			&p.PostOfficeName,
			&p.OfficeID,
			&p.OfficeName,
			&p.OfficeTypeCode,
			&p.Pincode,
			&p.DivisionName,
			&p.SubdivisionName,
			&p.CircleName,
			&p.RegionName,
		); err != nil {
			return nil, 0, err
		}

		result = append(result, p)
	}

	if err := rows.Err(); err != nil {
		return nil, 0, err
	}

	// Count query mirrors filters
	countQuery := `
SELECT COUNT(DISTINCT CONCAT(pmm.group_name, '-', pmm.cadre_name)) 
FROM pmdm.post_management_master pmm
LEFT JOIN pmdm.kafka_office_hierarchy koh
	ON pmm.office_id = koh.office_id
WHERE
	pmm.status = 'Active'
	AND  pmm.cadre_name = $1 
	and koh.circle_office_id = $2
`

	var total int
	if err := r.db.QueryRow(ctx, countQuery, cadreName, circleOfficeID).Scan(&total); err != nil {
		return nil, 0, err
	}

	return result, total, nil
}

func (r *PostManagementRepository) GetCircleSummaryRepo(
	gctx *gin.Context,
	cadreName string,
	circleOfficeID int64,
	regionOfficeID int64,
	includeList bool,
	meta port.MetaDataRequest,
) (*domain.CircleSummaryResponse, error) {

	ctx, cancel := context.WithTimeout(
		gctx.Request.Context(),
		r.cfg.GetDuration(DBtimeout),
	)
	defer cancel()

	if includeList {
		list, total, err := r.GetCircleSummaryWithList(
			ctx,
			cadreName,
			circleOfficeID,
			regionOfficeID,
			meta,
		)
		if err != nil {
			return nil, err
		}

		return &domain.CircleSummaryResponse{
			List:  list,
			Total: total,
		}, nil
	}

	summary, hier, total, err := r.GetCircleSummary1(
		ctx,
		cadreName,
		circleOfficeID,
		meta,
	)
	if err != nil {
		return nil, err
	}

	return &domain.CircleSummaryResponse{
		Summary:   summary,
		Hierarchy: hier,
		Total:     total,
	}, nil
}

func (r *PostManagementRepository) GetCircleSummary1(
	ctx context.Context,
	cadreName string,
	circleOfficeID int64,
	meta port.MetaDataRequest,
) ([]domain.CircleSummaryDetail, domain.CircleHierarchy, int, error) {

	query := `
	select
	koh.region_name,
	koh.region_office_id,
	koh.circle_name,
	koh.circle_office_id,
	COALESCE(pmm.group_name, '') as group_name,
	pmm.cadre_name,
	COUNT(*) as total_posts,
	COUNT(case when pmm.filled_status = 'Filled' then 1 end) as total_filled_posts,
	COUNT(case when pmm.filled_status = 'Vacant' or pmm.filled_status is null then 1 end) as total_vacant_posts
from
	pmdm.post_management_master pmm
join pmdm.kafka_office_hierarchy koh on
	pmm.office_id = koh.office_id
	and koh.circle_office_id = $1
	and koh.region_office_id is not null
	and koh.region_office_id <> 0
where
	pmm.status = 'Active'
	and pmm.cadre_name = $2
group by
	koh.region_name,
	koh.region_office_id,
	koh.circle_name,
	koh.circle_office_id,
	pmm.group_name,
	pmm.cadre_name
order by
	koh.region_name,
	pmm.group_name,
	pmm.cadre_name
limit $3 offset $4
	`

	rows, err := r.db.Query(
		ctx,
		query,
		circleOfficeID,
		cadreName,
		meta.Limit,
		meta.Skip,
	)
	if err != nil {
		return nil, domain.CircleHierarchy{}, 0, err
	}
	defer rows.Close()

	var result []domain.CircleSummaryDetail

	for rows.Next() {
		var cs domain.CircleSummaryDetail
		if err := rows.Scan(
			&cs.RegionName,
			&cs.RegionOfficeID,
			&cs.CircleName,
			&cs.CircleOfficeID,
			&cs.GroupName,
			&cs.CadreName,
			&cs.TotalPosts,
			&cs.TotalFilledPosts,
			&cs.TotalVacantPosts,
		); err != nil {
			return nil, domain.CircleHierarchy{}, 0, err
		}
		result = append(result, cs)
	}

	countQuery := `
	select
	COUNT(distinct koh.region_office_id) as total_count
from
	pmdm.post_management_master pmm
join pmdm.kafka_office_hierarchy koh on
	pmm.office_id = koh.office_id
	and koh.circle_office_id = $1
	and koh.region_office_id is not null
	and koh.region_office_id <> 0
where
	pmm.status = 'Active'
	and pmm.cadre_name = $2
	`

	var total int
	if err := r.db.QueryRow(ctx, countQuery, circleOfficeID, cadreName).Scan(&total); err != nil {
		return nil, domain.CircleHierarchy{}, 0, err
	}

	hierarchyQuery := `
	select
	distinct circle_name,
	circle_code
from
	pmdm.kafka_office_hierarchy
where
	circle_office_id = $1
limit 1
	`

	var hierarchy domain.CircleHierarchy
	if err := r.db.QueryRow(
		ctx,
		hierarchyQuery,
		circleOfficeID,
	).Scan(
		&hierarchy.CircleName,
		&hierarchy.CircleCode,
	); err != nil {
		return nil, domain.CircleHierarchy{}, 0, err
	}

	return result, hierarchy, total, nil
}

func (r *PostManagementRepository) GetCircleSummaryWithList(
	ctx context.Context,
	cadreName string,
	circleOfficeID int64,
	regionOfficeID int64,
	meta port.MetaDataRequest,
) ([]domain.PostSummaryDetail, int, error) {

	query := `
	SELECT
		pmm.postmanagement_id,
		COALESCE(pmm.post_name, '') as post_name,
		COALESCE(pmm.designation, '') as designation,
		COALESCE(pmm.filled_status, '') as filled_status,
		COALESCE(pmm.post_status, '') as post_status,
		COALESCE(pmm.pay_level, 0) as pay_level,
		COALESCE(pmm.grade_pay, 0) as grade_pay,
		COALESCE(pmm.sanctioned_strength, 0) as sanctioned_strength,
		COALESCE(pmm.permanent_status, false) as permanent_status,
		COALESCE(pmm.allowances_attached, false) as allowances_attached,
		COALESCE(pmm.allowance_description, '') as allowance_description,
		COALESCE(pmm.group_name, '') as group_name,
		COALESCE(pmm.cadre_name, '') as cadre_name,
		COALESCE(kom.office_name, '') as office_name,
		COALESCE(kom.office_id, 0) as office_id,
		COALESCE(kom.office_type_code, '') as office_type_code,
		COALESCE(kom.pincode, 0) as pincode,
		COALESCE(koh.division_name, '') as division_name,
		COALESCE(koh.subdivision_name, '') as subdivision_name,
		COALESCE(koh.circle_name, '') as circle_name,
		COALESCE(koh.region_name, '') as region_name
	from
	pmdm.post_management_master pmm
left join pmdm.kafka_office_hierarchy koh on
	pmm.office_id = koh.office_id
left join pmdm.kafka_office_master kom on
	pmm.office_id = kom.office_id
where
	pmm.status = 'Active'
	and pmm.cadre_name = $1
	and koh.circle_office_id = $2
	and koh.region_office_id = $3
order by
	kom.office_name,
	pmm.post_name,
	pmm.designation
	`

	rows, err := r.db.Query(
		ctx,
		query,
		cadreName,
		circleOfficeID,
		regionOfficeID,
	)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()

	var result []domain.PostSummaryDetail

	for rows.Next() {
		var p domain.PostSummaryDetail
		if err := rows.Scan(
			&p.PostManagementID,
			&p.PostName,
			&p.Designation,
			&p.FilledStatus,
			&p.PostStatus,
			&p.PayLevel,
			&p.GradePay,
			&p.SanctionedStrength,
			&p.PermanentStatus,
			&p.AllowancesAttached,
			&p.AllowanceDescription,
			&p.GroupName,
			&p.CadreName,
			&p.OfficeName,
			&p.OfficeID,
			&p.OfficeTypeCode,
			&p.Pincode,
			&p.DivisionName,
			&p.SubdivisionName,
			&p.CircleName,
			&p.RegionName,
		); err != nil {
			return nil, 0, err
		}
		result = append(result, p)
	}

	countQuery := `
	SELECT COUNT(*)
	FROM pmdm.post_management_master pmm
	JOIN pmdm.kafka_office_hierarchy koh
		ON pmm.office_id = koh.office_id
	WHERE pmm.status = 'Active'
	and pmm.cadre_name = $1
	and koh.circle_office_id = $2
	and koh.region_office_id = $3
	`

	var total int
	if err := r.db.QueryRow(ctx, countQuery, cadreName, circleOfficeID, regionOfficeID).Scan(&total); err != nil {
		return nil, 0, err
	}

	return result, total, nil
}

func (r *PostManagementRepository) GetRegionSummaryRepo(
	gctx *gin.Context,
	cadreName string,
	divisionOfficeID int64,
	regionOfficeID int64,
	includeList bool,
	meta port.MetaDataRequest,
) (*domain.RegionSummaryResponse, error) {

	ctx, cancel := context.WithTimeout(
		gctx.Request.Context(),
		r.cfg.GetDuration(DBtimeout),
	)
	defer cancel()

	if includeList {
		list, total, err := r.GetRegionSummaryWithList(
			ctx,
			cadreName,
			divisionOfficeID,
			regionOfficeID,
			meta,
		)
		if err != nil {
			return nil, err
		}

		return &domain.RegionSummaryResponse{
			List:  list,
			Total: total,
		}, nil
	}

	summary, hier, total, err := r.GetRegionSummary(
		ctx,
		cadreName,
		regionOfficeID,
		meta,
	)
	if err != nil {
		return nil, err
	}

	return &domain.RegionSummaryResponse{
		Summary:   summary,
		Total:     total,
		Hierarchy: hier,
	}, nil
}

func (r *PostManagementRepository) GetRegionSummary(
	ctx context.Context,
	cadreName string,
	regionOfficeID int64,
	meta port.MetaDataRequest,
) ([]domain.RegionSummaryDetail, domain.RegionHierarchy, int, error) {

	query := `
select
	koh.division_name,
	koh.division_office_id,
	koh.region_name,
	koh.region_office_id,
	COALESCE(pmm.group_name, '') as group_name,
	pmm.cadre_name,
	COUNT(*) as total_posts,
	COUNT(case when pmm.filled_status = 'Filled' then 1 end) as total_filled_posts,
	COUNT(case when pmm.filled_status = 'Vacant' or pmm.filled_status is null then 1 end) as total_vacant_posts
from
	pmdm.post_management_master pmm
join pmdm.kafka_office_hierarchy koh on
	pmm.office_id = koh.office_id
	and koh.region_office_id = $1
	and koh.division_office_id is not null
	and koh.division_office_id <> 0
where
	pmm.status = 'Active'
	and pmm.cadre_name = $2
group by
	koh.division_name,
	koh.division_office_id,
	koh.region_name,
	koh.region_office_id,
	pmm.group_name,
	pmm.cadre_name
order by
	koh.division_name,
	pmm.group_name,
	pmm.cadre_name
limit $3 offset $4
	`

	rows, err := r.db.Query(
		ctx,
		query,
		regionOfficeID,
		cadreName,
		meta.Limit,
		meta.Skip,
	)
	if err != nil {
		return nil, domain.RegionHierarchy{}, 0, err
	}
	defer rows.Close()

	var result []domain.RegionSummaryDetail

	for rows.Next() {
		var rs domain.RegionSummaryDetail
		if err := rows.Scan(
			&rs.DivisionName,
			&rs.DivisionOfficeID,
			&rs.RegionName,
			&rs.RegionOfficeID,
			&rs.GroupName,
			&rs.CadreName,
			&rs.TotalPosts,
			&rs.TotalFilledPosts,
			&rs.TotalVacantPosts,
		); err != nil {
			return nil, domain.RegionHierarchy{}, 0, err
		}
		result = append(result, rs)
	}

	countQuery := `
	select
	COUNT(distinct koh.division_office_id) as total_count
from
	pmdm.post_management_master pmm
join pmdm.kafka_office_hierarchy koh on
	pmm.office_id = koh.office_id
	and koh.region_office_id = $1
	and koh.division_office_id is not null
	and koh.division_office_id <> 0
where
	pmm.status = 'Active'
	and pmm.cadre_name = $2
	`

	var total int
	if err := r.db.QueryRow(ctx, countQuery, regionOfficeID, cadreName).Scan(&total); err != nil {
		return nil, domain.RegionHierarchy{}, 0, err
	}

	hierarchyQuery := `
	select
	distinct region_name,
	circle_name,
	circle_office_id
from
	pmdm.kafka_office_hierarchy
where
	region_office_id = $1
limit 1
	`

	var hierarchy domain.RegionHierarchy
	if err := r.db.QueryRow(
		ctx,
		hierarchyQuery,
		regionOfficeID,
	).Scan(
		&hierarchy.RegionName,
		&hierarchy.CircleName,
		&hierarchy.CircleOfficeID,
	); err != nil {
		return nil, domain.RegionHierarchy{}, 0, err
	}

	return result, hierarchy, total, nil
}

func (r *PostManagementRepository) GetRegionSummaryWithList(
	ctx context.Context,
	cadreName string,
	divisionOfficeID int64,
	regionOfficeID int64,
	meta port.MetaDataRequest,
) ([]domain.PostSummaryDetail, int, error) {

	query := `
	SELECT
		pmm.postmanagement_id,
		COALESCE(pmm.post_name, '') as post_name,
		COALESCE(pmm.designation, '') as designation,
		COALESCE(pmm.filled_status, '') as filled_status,
		COALESCE(pmm.post_status, '') as post_status,
		COALESCE(pmm.pay_level, 0) as pay_level,
		COALESCE(pmm.grade_pay, 0) as grade_pay,
		COALESCE(pmm.sanctioned_strength, 0) as sanctioned_strength,
		COALESCE(pmm.permanent_status, false) as permanent_status,
		COALESCE(pmm.allowances_attached, false) as allowances_attached,
		COALESCE(pmm.allowance_description, '') as allowance_description,
		COALESCE(pmm.group_name, '') as group_name,
		COALESCE(pmm.cadre_name, '') as cadre_name,
		COALESCE(kom.office_name, '') as office_name,
		COALESCE(kom.office_id, 0) as office_id,
		COALESCE(kom.office_type_code, '') as office_type_code,
		COALESCE(kom.pincode, 0) as pincode,
		COALESCE(koh.division_name, '') as division_name,
		COALESCE(koh.subdivision_name, '') as subdivision_name,
		COALESCE(koh.circle_name, '') as circle_name,
		COALESCE(koh.region_name, '') as region_name
from
	pmdm.post_management_master pmm
left join pmdm.kafka_office_hierarchy koh on
	pmm.office_id = koh.office_id
left join pmdm.kafka_office_master kom on
	pmm.office_id = kom.office_id
where
	pmm.status = 'Active'
	and pmm.cadre_name = $1
	and koh.region_office_id = $2
	and koh.division_office_id = $3
order by
	kom.office_name,
	pmm.post_name,
	pmm.designation
	`

	rows, err := r.db.Query(
		ctx,
		query,
		cadreName,
		regionOfficeID,
		divisionOfficeID,
	)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()

	var result []domain.PostSummaryDetail

	for rows.Next() {
		var p domain.PostSummaryDetail
		if err := rows.Scan(
			&p.PostManagementID,
			&p.PostName,
			&p.Designation,
			&p.FilledStatus,
			&p.PostStatus,
			&p.PayLevel,
			&p.GradePay,
			&p.SanctionedStrength,
			&p.PermanentStatus,
			&p.AllowancesAttached,
			&p.AllowanceDescription,
			&p.GroupName,
			&p.CadreName,
			&p.OfficeName,
			&p.OfficeID,
			&p.OfficeTypeCode,
			&p.Pincode,
			&p.DivisionName,
			&p.SubdivisionName,
			&p.CircleName,
			&p.RegionName,
		); err != nil {
			return nil, 0, err
		}
		result = append(result, p)
	}

	countQuery := `
	SELECT COUNT(*)
	FROM pmdm.post_management_master pmm
	JOIN pmdm.kafka_office_hierarchy koh
		ON pmm.office_id = koh.office_id
	WHERE pmm.status = 'Active'
	  AND pmm.cadre_name = $1
	  AND koh.division_office_id = $2
	  AND koh.region_office_id = $3
	`

	var total int
	if err := r.db.QueryRow(
		ctx,
		countQuery,
		cadreName,
		divisionOfficeID,
		regionOfficeID,
	).Scan(&total); err != nil {
		return nil, 0, err
	}

	return result, total, nil
}

func (r *PostManagementRepository) GetDivisionSummaryRepo(
	gctx *gin.Context,
	cadreName string,
	OfficeID int64,
	divisionOfficeID int64,
	includeList bool,
	meta port.MetaDataRequest,
) (*domain.DivisionSummaryResponse, error) {

	ctx, cancel := context.WithTimeout(
		gctx.Request.Context(),
		r.cfg.GetDuration(DBtimeout),
	)
	defer cancel()

	if includeList {
		list, total, err := r.GetDivisionSummaryWithList(
			ctx,
			cadreName,
			OfficeID,
			divisionOfficeID,
			meta,
		)
		if err != nil {
			return nil, err
		}

		return &domain.DivisionSummaryResponse{
			List:  list,
			Total: total,
		}, nil
	}

	summary, heir, total, err := r.GetDivisionSummary(
		ctx,
		divisionOfficeID,
		cadreName,
		meta,
	)
	if err != nil {
		return nil, err
	}

	return &domain.DivisionSummaryResponse{
		Summary:   summary,
		Hierarchy: *heir,
		Total:     total,
	}, nil
}

func (r *PostManagementRepository) GetDivisionSummary(
	ctx context.Context,
	divisionOfficeID int64,
	cadreName string,
	meta port.MetaDataRequest,
) ([]domain.DivisionSummaryDetail, *domain.DivisionHierarchyInfo, int, error) {

	query := `
	SELECT
		kom.office_id,
		kom.office_name,
		kom.office_type_code,
		kom.pincode,
		COALESCE(kom.email_id, ''),
		COALESCE(kom.contact_number, ''),
		koh.division_name,
		koh.division_office_id,
		COALESCE(koh.subdivision_name, ''),
		COALESCE(pmm.group_name, '') as group_name,
		pmm.cadre_name,
		COUNT(*) AS total_posts,
		COUNT(CASE WHEN pmm.filled_status = 'Filled' THEN 1 END) AS total_filled_posts,
		COUNT(CASE WHEN pmm.filled_status = 'Vacant' OR pmm.filled_status IS NULL THEN 1 END) AS total_vacant_posts
	FROM pmdm.post_management_master pmm
	JOIN pmdm.kafka_office_hierarchy koh
		ON pmm.office_id = koh.office_id
		AND koh.division_office_id = $1
	JOIN pmdm.kafka_office_master kom
		ON koh.office_id = kom.office_id
	WHERE pmm.status = 'Active'
	  AND pmm.cadre_name = $2
	GROUP BY
		kom.office_id,
		kom.office_name,
		kom.office_type_code,
		kom.pincode,
		kom.email_id,
		kom.contact_number,
		koh.division_name,
		koh.division_office_id,
		koh.subdivision_name,
		pmm.group_name,
		pmm.cadre_name
	ORDER BY
		kom.office_name,
		pmm.group_name,
		pmm.cadre_name
	LIMIT $3 OFFSET $4
	`

	rows, err := r.db.Query(
		ctx,
		query,
		divisionOfficeID,
		cadreName,
		meta.Limit,
		meta.Skip,
	)
	if err != nil {
		return nil, nil, 0, err
	}
	defer rows.Close()

	var result []domain.DivisionSummaryDetail

	for rows.Next() {
		var d domain.DivisionSummaryDetail
		if err := rows.Scan(
			&d.OfficeId,
			&d.OfficeName,
			&d.OfficeTypeCode,
			&d.Pincode,
			&d.Email,
			&d.ContactNumber,
			&d.DivisionName,
			&d.DivisionOfficeID,
			&d.SubDivisionName,
			&d.GroupName,
			&d.CadreName,
			&d.TotalPosts,
			&d.TotalFilledPosts,
			&d.TotalVacantPosts,
		); err != nil {
			return nil, nil, 0, err
		}
		result = append(result, d)
	}

	// -----------------------------
	// 2. Count query
	// -----------------------------
	countQuery := `
	SELECT COUNT(DISTINCT koh.office_id)
	FROM pmdm.post_management_master pmm
	JOIN pmdm.kafka_office_hierarchy koh
		ON pmm.office_id = koh.office_id
		AND koh.division_office_id = $1
	JOIN pmdm.kafka_office_master kom
		ON koh.office_id = kom.office_id
	WHERE pmm.status = 'Active'
	  AND pmm.cadre_name = $2
	`

	var total int
	if err := r.db.QueryRow(
		ctx,
		countQuery,
		divisionOfficeID,
		cadreName,
	).Scan(&total); err != nil {
		return nil, nil, 0, err
	}

	// -----------------------------
	// 3. Division hierarchy info
	// -----------------------------
	hierarchyQuery := `
	SELECT
		division_name,
		region_name,
		region_office_id,
		circle_name,
		circle_office_id
	FROM pmdm.kafka_office_hierarchy
	WHERE division_office_id = $1
	LIMIT 1
	`

	var hierarchy domain.DivisionHierarchyInfo
	if err := r.db.QueryRow(
		ctx,
		hierarchyQuery,
		divisionOfficeID,
	).Scan(
		&hierarchy.DivisionName,
		&hierarchy.RegionName,
		&hierarchy.RegionOfficeID,
		&hierarchy.CircleName,
		&hierarchy.CircleOfficeID,
	); err != nil {
		return nil, nil, 0, err
	}

	return result, &hierarchy, total, nil
}

func (r *PostManagementRepository) GetDivisionSummaryWithList(
	ctx context.Context,
	cadreName string,
	OfficeID int64,
	divisionOfficeID int64,
	meta port.MetaDataRequest,
) ([]domain.PostSummaryDetail, int, error) {

	query := `
	SELECT
		pmm.postmanagement_id,
		COALESCE(pmm.post_name, ''),
		COALESCE(pmm.designation, ''),
		COALESCE(pmm.filled_status, ''),
		COALESCE(pmm.post_status, ''),
		COALESCE(pmm.pay_level, 0),
		COALESCE(pmm.grade_pay, 0),
		COALESCE(pmm.sanctioned_strength, 0),
		COALESCE(pmm.permanent_status, false),
		COALESCE(pmm.allowances_attached, false),
		COALESCE(pmm.allowance_description, ''),
		COALESCE(pmm.group_name, ''),
		COALESCE(pmm.cadre_name, ''),
		COALESCE(kom.office_name, ''),
		COALESCE(kom.office_id, 0),
		COALESCE(kom.office_type_code, ''),
		COALESCE(kom.pincode, 0),
		COALESCE(koh.division_name, ''),
		COALESCE(koh.subdivision_name, ''),
		COALESCE(koh.circle_name, ''),
		COALESCE(koh.region_name, '')
from
	pmdm.post_management_master pmm
left join pmdm.kafka_office_hierarchy koh on
	pmm.office_id = koh.office_id
left join pmdm.kafka_office_master kom on
	pmm.office_id = kom.office_id
where
	pmm.status = 'Active'
	and pmm.cadre_name = $1
	and koh.division_office_id = $2
	and pmm.office_id = $3
order by
	pmm.post_name,
	pmm.designation
	`

	rows, err := r.db.Query(
		ctx,
		query,
		cadreName,
		divisionOfficeID,
		OfficeID,
	)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()

	var result []domain.PostSummaryDetail

	for rows.Next() {
		var p domain.PostSummaryDetail
		if err := rows.Scan(
			&p.PostManagementID,
			&p.PostName,
			&p.Designation,
			&p.FilledStatus,
			&p.PostStatus,
			&p.PayLevel,
			&p.GradePay,
			&p.SanctionedStrength,
			&p.PermanentStatus,
			&p.AllowancesAttached,
			&p.AllowanceDescription,
			&p.GroupName,
			&p.CadreName,
			&p.OfficeName,
			&p.OfficeID,
			&p.OfficeTypeCode,
			&p.Pincode,
			&p.DivisionName,
			&p.SubdivisionName,
			&p.CircleName,
			&p.RegionName,
		); err != nil {
			return nil, 0, err
		}
		result = append(result, p)
	}

	countQuery := `
	SELECT COUNT(*)
	FROM pmdm.post_management_master pmm
	JOIN pmdm.kafka_office_hierarchy koh
		ON pmm.office_id = koh.office_id
	WHERE pmm.status = 'Active'
	  AND pmm.cadre_name = $1
	  AND pmm.office_id = $2
	  AND koh.division_office_id = $3
	`

	var total int
	if err := r.db.QueryRow(
		ctx,
		countQuery,
		cadreName,
		OfficeID,
		divisionOfficeID,
	).Scan(&total); err != nil {
		return nil, 0, err
	}

	return result, total, nil
}

func (r *PostManagementRepository) GetPostSummaryRepo1(
	gctx *gin.Context,
	cadreName string,
	includeList bool,
	search string,
	meta port.MetaDataRequest,
) (*domain.PostSummaryResponse1, error) {

	ctx, cancel := context.WithTimeout(
		gctx.Request.Context(),
		r.cfg.GetDuration(DBtimeout),
	)
	defer cancel()

	if includeList {
		list, total, err := r.GetPostSummaryWithList(
			ctx,
			cadreName,
			search,
			meta,
		)
		if err != nil {
			return nil, err
		}

		return &domain.PostSummaryResponse1{
			List:  list,
			Total: total,
		}, nil
	}

	summary, total, err := r.GetPostSummary(
		ctx,
		cadreName,
		search,
		meta,
	)
	if err != nil {
		return nil, err
	}

	return &domain.PostSummaryResponse1{
		Summary: summary,
		Total:   total,
	}, nil
}

func (r *PostManagementRepository) GetPostSummary(
	ctx context.Context,
	cadreName string,
	search string,
	meta port.MetaDataRequest,
) ([]domain.PostSummaryDetail1, int, error) {

	query := `
SELECT
    COALESCE(pmm.group_name, '')        AS group_name,
    COALESCE(pmm.cadre_name, '')        AS cadre_name,
    COUNT(*)                            AS total_posts,
    COUNT(CASE WHEN pmm.filled_status = 'Filled' THEN 1 END) AS total_filled_posts,
    COUNT(CASE WHEN pmm.filled_status = 'Vacant'
                OR pmm.filled_status IS NULL THEN 1 END) AS total_vacant_posts
FROM pmdm.post_management_master pmm
JOIN pmdm.kafka_office_hierarchy koh
    ON pmm.office_id = koh.office_id
WHERE pmm.status = 'Active'
GROUP BY
    COALESCE(pmm.group_name, ''),
    COALESCE(pmm.cadre_name, '')
ORDER BY
    group_name,
    cadre_name
LIMIT $1 OFFSET $2;
	`

	rows, err := r.db.Query(
		ctx,
		query,
		meta.Limit,
		meta.Skip,
	)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()

	var result []domain.PostSummaryDetail1

	for rows.Next() {
		var cs domain.PostSummaryDetail1
		if err := rows.Scan(
			&cs.GroupName,
			&cs.CadreName,
			&cs.TotalPosts,
			&cs.TotalFilledPosts,
			&cs.TotalVacantPosts,
		); err != nil {
			return nil, 0, err
		}
		result = append(result, cs)
	}

	countQuery := `
	select
	COUNT(distinct CONCAT(pmm.group_name, '-', pmm.cadre_name)) as total_count
from
	pmdm.post_management_master pmm
join pmdm.kafka_office_hierarchy koh on
	pmm.office_id = koh.office_id
where
	pmm.status = 'Active'
	`

	var total int
	if err := r.db.QueryRow(ctx, countQuery).Scan(&total); err != nil {
		return nil, 0, err
	}

	return result, total, nil
}

func (r *PostManagementRepository) GetPostSummaryWithList(
	ctx context.Context,
	cadreName string,
	search string,
	meta port.MetaDataRequest,
) ([]domain.PostSummaryDetail, int, error) {

	query := `
	SELECT
		pmm.postmanagement_id,
		COALESCE(pmm.post_name, ''),
		COALESCE(pmm.designation, ''),
		COALESCE(pmm.filled_status, ''),
		COALESCE(pmm.post_status, ''),
		COALESCE(pmm.pay_level, 0),
		COALESCE(pmm.grade_pay, 0),
		COALESCE(pmm.sanctioned_strength, 0),
		COALESCE(pmm.permanent_status, false),
		COALESCE(pmm.allowances_attached, false),
		COALESCE(pmm.allowance_description, ''),
		COALESCE(pmm.group_name, ''),
		COALESCE(pmm.cadre_name, ''),
		COALESCE(pmm.office_name, ''),
		COALESCE(pmm.office_id, 0)
from
	pmdm.post_management_master pmm
left join pmdm.kafka_office_hierarchy koh on
	pmm.office_id = koh.office_id
left join pmdm.kafka_office_master kom on
	pmm.office_id = kom.office_id
where
	pmm.status = 'Active'
	and pmm.cadre_name = $1
order by
	kom.office_name,
	pmm.post_name,
	pmm.designation
	`

	rows, err := r.db.Query(
		ctx,
		query,
		cadreName,
	)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()

	var result []domain.PostSummaryDetail

	for rows.Next() {
		var p domain.PostSummaryDetail
		if err := rows.Scan(
			&p.PostManagementID,
			&p.PostName,
			&p.Designation,
			&p.FilledStatus,
			&p.PostStatus,
			&p.PayLevel,
			&p.GradePay,
			&p.SanctionedStrength,
			&p.PermanentStatus,
			&p.AllowancesAttached,
			&p.AllowanceDescription,
			&p.GroupName,
			&p.CadreName,
			&p.PostOfficeName,
			&p.OfficeID,
		); err != nil {
			return nil, 0, err
		}
		result = append(result, p)
	}

	countQuery := `
	SELECT COUNT(*)
	FROM pmdm.post_management_master
	WHERE status = 'Active'
	  AND ($1 = '' OR cadre_name = $1)
	`

	var total int
	if err := r.db.QueryRow(ctx, countQuery, cadreName).Scan(&total); err != nil {
		return nil, 0, err
	}

	return result, total, nil
}

func (pmr *PostManagementRepository) FetchAllPostByOfficeID2(gctx *gin.Context, officeID int) ([]domain.PostManagementMasterNew, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.
		Select(
			"COALESCE(pm.office_id, 0) AS office_id",
			"COALESCE(pm.cadre_id, 0) AS cadre_id",
			"COALESCE(pm.cadre_name, '') AS cadre_name",
			"COALESCE(pm.group_id, 0) AS group_id",
			"COALESCE(pm.group_name, '') AS group_name",
			"COALESCE(pm.employee_group, '') AS employee_group",
			"COALESCE(pm.post_id, 0) AS post_id",
			"COALESCE(pm.post_name, '') AS post_name",
			"COALESCE(pm.designation_id, 0) AS designation_id",
			"COALESCE(pm.designation, '') AS designation",
			`CASE 
			WHEN km.employee_id IS NULL OR km.employment_status != 'Active' 
			THEN 'Vacant'
			ELSE 'Filled'
		END AS filled_status`,
			"COALESCE(kom.office_name, '') AS office_name",
			"COALESCE(km.post_status, '') AS post_status",
			"COALESCE(km.employee_id, 0) AS employee_id",
			"COALESCE(CONCAT(km.employee_first_name, ' ', km.employee_middle_name, ' ', km.employee_last_name), '') AS employee_name",
			"COALESCE(pm.is_head_of_the_office, false) AS is_head_of_the_office",
			"COALESCE(pmd.employee_post_id, 0) AS employee_post_id",
			"COALESCE(pmd.leave_sanc_authority_1, 0) AS leave_sanc_authority_1",
			"COALESCE(pmd.leave_sanc_authority_2, 0) AS leave_sanc_authority_2",
			"COALESCE(pmd.pay_approve_authority1, 0) AS pay_approve_authority1",
			"COALESCE(pmd.appointing_authority, 0) AS appointing_authority",
			"COALESCE(pmd.disciplinary_authority, 0) AS disciplinary_authority",
			"COALESCE(pmd.ddo_authority, 0) AS ddo_authority",
			"COALESCE(pmd.employee_office_id, 0) AS employee_office_id",
			"COALESCE(pmd.vigilence_maker_authority, 0) AS vigilence_maker_authority",
			"COALESCE(pm.permanent_status, false) AS permanent_status",
			"COALESCE(pm.allowances_attached, false) AS allowances_attached",
			"COALESCE(pm.allowance_description, '') AS allowance_description",
			"pm.status",
			"pm.created_by",
			"COALESCE(pm.created_date, '1970-01-01'::timestamp) AS created_date",
			"COALESCE(pm.approved_by, '') AS approved_by",
			"COALESCE(pm.approved_date, '1970-01-01'::timestamp) AS approved_date",
			"COALESCE(pm.updated_by, '') AS updated_by",
			"COALESCE(pm.updated_date, '1970-01-01'::timestamp) AS updated_date",
			"COALESCE(pm.valid_from, '1970-01-01'::timestamp) AS valid_from",
			"COALESCE(pm.valid_to, '1970-01-01'::timestamp) AS valid_to",
			"COALESCE(pm.order_casemark, '') AS order_casemark",
			"COALESCE(pm.order_date, '1970-01-01'::timestamp) AS order_date",
			"COALESCE(pm.upload_order_doc_name, '') AS upload_order_doc_name",
			"COALESCE(pm.establishment_register_id, 0) AS establishment_register_id",
			"COALESCE(pm.establishment_register_name, '') AS establishment_register_name",
			"COALESCE(pm.remarks, '') AS remarks",
		).
		From("pmdm.post_management_master pm").
		LeftJoin("pmdm.kafka_employee_master km ON km.post_id = pm.post_id AND km.employment_status = 'Active'").
		LeftJoin("pmdm.post_mapping_detail pmd ON pmd.employee_post_id = pm.post_id").
		LeftJoin("pmdm.kafka_office_master kom ON kom.office_id = pm.office_id").
		Where(sq.And{
			sq.Eq{"pm.office_id": officeID},
			//sq.Eq{"pm.status": "Active"},
			//sq.Eq{"pm.cadre_id": cadreId},
		})

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMasterNew])
}

func (pmr *PostManagementRepository) PostManagementByOfficeIDQueryMDWD1(gctx *gin.Context, officeID int, raPostID int64, reqMetadata port.MetaDataRequest) ([]domain.PostManagementMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select(QOfficeID, QPostName, QOfficeName, QGroupID, QFilledStatus, QPostID, QDestination, QPermanentStatus, QStatus).
		From(PMPostManagementMaster).
		Join("pmdm.post_mapping_detail pmd on pmd.employee_post_id=pm.post_id and pmd.role_authority=" + strconv.Itoa(int(raPostID))).
		Where(sq.Eq{"employee_office_id": officeID}).
		Where(sq.Eq{"status": "Active"}).
		Where("pm.updated_date = CURRENT_DATE - INTERVAL '1 day'").
		OrderBy("office_id").
		Offset(uint64(reqMetadata.Skip * reqMetadata.Limit)).
		Limit(uint64(reqMetadata.Limit))
	//Where(sq.Eq{"office_id": officeID})
	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaster])
}

func (pmr *PostManagementRepository) PostManagementByCadreAndOfficeQueryD1(gctx *gin.Context, cadreName string, officeID int, reqMetadata port.MetaDataRequest) ([]domain.PostManagementMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select("office_id", "post_id", "post_name", "cadre_name", "post_status", "filled_status", "status").
		From(PostManagementMaster).
		Where(sq.Eq{"cadre_name": cadreName, "office_id": officeID}).
		Where("updated_date = CURRENT_DATE - INTERVAL '1 day'").
		OrderBy("office_id").
		Offset(uint64(reqMetadata.Skip * reqMetadata.Limit)).
		Limit(uint64(reqMetadata.Limit))

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaster])
}

func (pmr *PostManagementRepository) PostManagementByOfficeIDQueryD1(gctx *gin.Context, validFrom, validTo time.Time, metadata port.MetaDataRequest) ([]domain.PostManagementMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()
	//log.Error(ctx,ContextWithTimeout)

	query := dblib.Psql.Select(
		PMOfficeID, // assuming office_id is an integer, use 0 as default
		PMPostName,
		PMOfficeName,
		PMGroupID, // assuming group_id is an integer, use 0 as default
		PMFilledStatus,
		PMPostID, // assuming post_id is an integer, use 0 as default
		PMDestinaton,
		PMPermanentStatus,
		PMStatus,
		PMGroupName,
		PMEmployeeGroup,
		PMCradeID, // assuming cadre_id is an integer, use 0 as default
		PMCradeName,
		"COALESCE(pay_level, 0) AS pay_level", // assuming pay_level is an integer, use 0 as default
		"COALESCE(grade_pay, 0) AS grade_pay", // assuming grade_pay is an integer, use 0 as default
		"COALESCE(designation_id, 0) AS designation_id", // assuming designation_id is an integer, use 0 as default
	).
		From(PMPostManagementMaster).
		Where(sq.Eq{"status": "Active"}).
		Where(sq.And{
			sq.GtOrEq{"updated_date": validFrom},
			sq.LtOrEq{"updated_date": validTo},
		}).
		Offset(uint64(metadata.Skip * metadata.Limit)).
		Limit(uint64(metadata.Limit))

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMaster])
}

func (pmr *PostManagementRepository) GetPostDetailByOfficeID(
	gctx *gin.Context,
	officeID int64,
	reqMetadata port.MetaDataRequest,
) ([]domain.PostDetails1, error) {

	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.
		Select(
			"DISTINCT ON (pmm.post_id) pmm.post_name",
			"pmm.post_id",
			"pmm.group_id",
			"pmm.designation",
			"pmm.status as post_status",
			"kem.post_status as filled_status",
		).
		From("pmdm.post_management_master pmm").
		Join("pmdm.kafka_employee_master kem ON kem.post_id = pmm.post_id").
		Where(sq.Eq{"pmm.office_id": officeID}).
		OrderBy(
			"pmm.post_id",
			`CASE 
				WHEN kem.post_status = 'Filled' THEN 1
				WHEN kem.post_status = 'Vacant' THEN 2
				ELSE 3
			END`,
		)

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostDetails1])
}
