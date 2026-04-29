package repo

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"path"
	"pmdm/core/domain"
	"pmdm/core/port"
	"sort"
	"strconv"
	"strings"
	"time"

	//"github.com/Masterminds/squirrel"
	sq "github.com/Masterminds/squirrel"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5"
	"github.com/minio/minio-go/v7"
	config "gitlab.cept.gov.in/it-2.0-common/api-config"
	dblib "gitlab.cept.gov.in/it-2.0-common/api-db"
	log "gitlab.cept.gov.in/it-2.0-common/api-log"
)

// CadreMasterRepository implements port.CadreMasterRepository interface
// and provides access to the postgres database for cadre master-related operations
type PosttoPostMappingRepository struct {
	db    *dblib.DB
	cfg   *config.Config
	minio *minio.Client
}

// NewCadreMasterRepository creates a new CadreMasterRepository instance
func NewPosttoPostMappingRepository(db *dblib.DB, cfg *config.Config, minio *minio.Client) *PosttoPostMappingRepository {
	return &PosttoPostMappingRepository{
		db,
		cfg,
		minio,
	}
}

func (ear *PosttoPostMappingRepository) GetMulAuthPostID(gctx *gin.Context, postid int, roleids []string, reqMetadata port.MetaDataRequest) ([]domain.PosttoPostMap, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), ear.cfg.GetDuration(DBtimeout))
	defer cancel()
	var empauthmasts []domain.PostMapMaster
	var empauths []domain.PosttoPostMap

	// Fetching authority column names for the provided role IDs
	query := dblib.Psql.Select("mapping_id , post_mapping_column_name").
		From("pmdm.post_mapping_master").
		Where(sq.Eq{"mapping_id": roleids, "post_mapping_status": "Active"}).
		OrderBy("mapping_id").
		Offset(uint64(reqMetadata.Skip * reqMetadata.Limit)).
		Limit(uint64(reqMetadata.Limit))

	empauthmasts, err := dblib.SelectRows(ctx, ear.db, query, pgx.RowToStructByNameLax[domain.PostMapMaster])
	if err != nil {
		return nil, err
	}
	var PostMapPostId, OfficeId int32
	var empauth domain.PosttoPostMap

	// Fetching authority mapping for each authority column name
	for _, mast := range empauthmasts {
		postmapquery := dblib.Psql.Select("employee_post_id", "COALESCE("+mast.PostMappingColumnName+", 0)").
			From("pmdm.post_mapping_detail").
			Where(sq.Eq{"employee_post_id": postid})

		sql1, args1, err1 := postmapquery.ToSql()
		if err1 != nil {
			log.Debug(gctx, "error at execute", err1)
			return nil, err1
		}

		err = ear.db.QueryRow(ctx, sql1, args1...).Scan(&postid,
			&PostMapPostId,
		)

		if err != nil {
			log.Debug(gctx, ErrorFor, err)
			return nil, err
		}

		query := dblib.Psql.Select("office_id").From("pmdm.post_management_master").Where(sq.Eq{"post_id": PostMapPostId})

		sql2, args2, err2 := query.ToSql()
		if err2 != nil {
			log.Debug(gctx, "error at execute", err1)
			return nil, err1
		}

		err = ear.db.QueryRow(ctx, sql2, args2...).Scan(&OfficeId)
		if err != nil {
			log.Debug(gctx, ErrorFor, err)
			return nil, err
		}
		empauth = domain.PosttoPostMap{
			EmployeePostID:        (postid),
			PostMapID:             mast.PostMapID,
			PostMappingColumnName: mast.PostMappingColumnName,
			PostMapPostId:         PostMapPostId,
			OfficeID:              int(OfficeId),
		}

		empauths = append(empauths, empauth)
	}

	return empauths, nil
}

// GetPostMappingMasterQuery retrieves post mapping master details
func (pmr *PosttoPostMappingRepository) GetPostMappingMasterQuery(gctx *gin.Context) ([]domain.PostMapMaster, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.Select("mapping_id", "post_mapping_column_name", "post_mapping_description").
		From("pmdm.post_mapping_master").
		Where("post_mapping_status = ?", "Active")

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostMapMaster])
}

func (pmr *PosttoPostMappingRepository) UpdateArrayOfEmpPostIDForParticularFieldQuery(gctx *gin.Context, empPostIDs []int, fieldName string, fieldValue interface{}, OfficeID int) ([]domain.PosttoPostMap, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	log.Debug(gctx, "Starting UpdateArrayOfEmpPostIDForParticularFieldQuery")

	// Validate the field name
	validFields := map[string]bool{
		"gds_leave_sanc_authority_1":      true,
		"gds_leave_sanc_authority_2":      true,
		"reporting_authority":             true,
		"apar_reporting_authority":        true,
		"apar_review_authority":           true,
		"apar_controlling_authority":      true,
		"apar_appellate_authority":        true,
		"service_book_approve_authority1": true,
		"leave_sanc_authority_1":          true,
		"leave_sanc_authority_2":          true,
		"leave_sanc_authority_3":          true,
		"pay_approve_authority1":          true,
		//"pay_approve_authority2":          true,
		"leave_fwd_authority1":            true,
		"leave_fwd_authority2":            true,
		"pay_fwd_authority1":              true,
		"pay_fwd_authority2":              true,
		"appointing_authority":            true,
		"disciplinary_authority":          true,
		"ddo_authority":                   true,
		"admin_authority":                 true,
		"pension_sanctioning_authority":   true,
		"pension_authorising_authority":   true,
		"service_book_approve_authority2": true,
		"role_authority":                  true,
		"service_book_foward_authority1":  true,
		"service_book_foward_authority2":  true,
		"apar_custodian_authority":        true,
	}

	if !validFields[fieldName] {
		log.Debug(gctx, "Invalid field name:", fieldName)
		return nil, errors.New("invalid field name")
	}
	log.Debug(gctx, "Field name validated:", fieldName)

	// Build the update query
	queryBuilder := sq.Update("pmdm.post_mapping_detail").
		Set(fieldName, fieldValue).
		Set("updated_date", time.Now()).
		Where(sq.Eq{"employee_post_id": empPostIDs}).
		Suffix("RETURNING employee_post_id, updated_date")

	sql, args, err := queryBuilder.PlaceholderFormat(sq.Dollar).ToSql() // Adjusted here
	if err != nil {
		log.Debug(gctx, "Error building SQL query:", err)
		return nil, err
	}
	log.Debug(gctx, "SQL query built successfully:", sql)

	// Execute the update query
	rows, err := pmr.db.Query(ctx, sql, args...)
	if err != nil {
		log.Debug(gctx, "Error executing SQL query:", err)
		return nil, err
	}
	defer rows.Close()

	var updatedResponses []domain.PosttoPostMap
	var updatedCount int
	for rows.Next() {
		var updatedResponse domain.PosttoPostMap
		if err := rows.Scan(&updatedResponse.EmployeePostID, &updatedResponse.UpdatedDate); err != nil {
			log.Debug(gctx, "Error scanning row:", err)
			return nil, err
		}
		updatedResponses = append(updatedResponses, updatedResponse)
		updatedCount++
	}
	if err := rows.Err(); err != nil {
		log.Debug(gctx, "Error in result set:", err)
		return nil, err
	}
	log.Debug(gctx, "Rows scanned and mapped successfully")

	// Check if any rows were updated
	if updatedCount == 0 {
		log.Debug(gctx, "No rows were updated, inserting new records")
		for _, empPostID := range empPostIDs {
			// Build the insert query
			insertQuery := fmt.Sprintf(`
                INSERT INTO pmdm.post_mapping_detail 
                (employee_post_id, employee_office_id, %s, updated_date) 
                VALUES ($1, $2, $3, $4) 
                ON CONFLICT DO NOTHING 
                RETURNING employee_post_id, updated_date
            `, fieldName)

			// Execute the insert query for the current empPostID
			rows, err := pmr.db.Query(ctx, insertQuery, empPostID, OfficeID, fieldValue, time.Now())
			if err != nil {
				log.Debug(gctx, "Error executing insert SQL query for empPostID:", empPostID, ":", err)
				return nil, err
			}

			defer rows.Close()

			// Check if any rows were affected
			var rowsAffected int
			for rows.Next() {
				rowsAffected++
				// Process the result rows if needed
			}

			if err := rows.Err(); err != nil {
				log.Error(gctx, "Error in result set:", err)
				return nil, err
			}

			if rowsAffected > 0 {
				log.Debug(gctx, "Rows affected for empPostID %d: %d\n", empPostID, rowsAffected)
			} else {
				log.Debug(gctx, "No rows affected for empPostID %d\n", empPostID)
			}
		}
	}
	log.Debug(gctx, "UpdateArrayOfEmpPostIDForParticularFieldQuery completed successfully")
	return updatedResponses, nil
}

func (ear *PosttoPostMappingRepository) GetAuthorityDetailsByPostID(gctx *gin.Context, postid int32, reqMetaData port.MetaDataRequest) (map[string]domain.AuthorityDetails, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), ear.cfg.GetDuration(DBtimeout))
	defer cancel()

	// Query the database to fetch authority details for the given employee_post_id
	var authority domain.PosttoPostMap
	err := ear.db.QueryRow(ctx, `
        SELECT
            COALESCE(gds_leave_sanc_authority_1, 0),
            COALESCE(gds_leave_sanc_authority_2, 0),
            COALESCE(reporting_authority, 0),
            COALESCE(apar_reporting_authority, 0),
            COALESCE(apar_review_authority, 0),
            COALESCE(apar_controlling_authority, 0),
            COALESCE(apar_appellate_authority, 0),
            COALESCE(service_book_approve_authority1, 0),
            COALESCE(leave_sanc_authority_1, 0),
            COALESCE(leave_sanc_authority_2, 0),
            COALESCE(leave_sanc_authority_3, 0),
            COALESCE(pay_approve_authority1, 0),
            
            COALESCE(leave_fwd_authority1, 0),
            COALESCE(leave_fwd_authority2, 0),
            COALESCE(pay_fwd_authority1, 0),
            COALESCE(pay_fwd_authority2, 0),
            COALESCE(appointing_authority, 0),
            COALESCE(disciplinary_authority, 0),
            COALESCE(ddo_authority, 0),
            COALESCE(admin_authority, 0),
            COALESCE(pension_sanctioning_authority, 0),
            COALESCE(pension_authorising_authority, 0),
            COALESCE(service_book_approve_authority2, 0),
            COALESCE(role_authority, 0),
            COALESCE(service_book_foward_authority1, 0),
            COALESCE(service_book_foward_authority2, 0),
			COALESCE(vigilence_maker_authority, 0),
			COALESCE(admin_office, 0),
			COALESCE(apar_custodian_authority,0)
        FROM pmdm.post_mapping_detail WHERE employee_post_id = $1
    `, postid).Scan(
		&authority.GDSLeaveSancAuthority1,
		&authority.GDSLeaveSancAuthority2,
		&authority.ReportingAuthority,
		&authority.AparReportingAuthority,
		&authority.AparReviewAuthority,
		&authority.AparAcceptingAuthority,
		&authority.AparRepresentAuthority,
		&authority.ServiceBookApproveAuthority1,
		&authority.LeaveSancAuthority1,
		&authority.LeaveSancAuthority2,
		&authority.LeaveSancAuthority3,
		&authority.PayApproveAuthority1,
		//&authority.PayApproveAuthority2,
		&authority.LeaveFWDAuthority1,
		&authority.LeaveFWDAuthority2,
		&authority.PayFWDAuthority1,
		&authority.PayFWDAuthority2,
		&authority.AppointingAuthority,
		&authority.DisciplinaryAuthority,
		&authority.DdoAuthority,
		&authority.AdminAuthority,
		&authority.PensionSanctioningAuthority,
		&authority.PensionAuthorisingAuthority,
		&authority.ServiceBookApproveAuthority2,
		&authority.RoleAuthority,
		&authority.ServiceBookForwardAuthority1,
		&authority.ServiceBookForwardAuthority2,
		&authority.VigilenceMakerAuthority,
		&authority.AdminOffice,
		&authority.AparCustodianAuhtority,
	)
	if err != nil {
		log.Debug(gctx, ErrorQueryAuthority, err)
		return nil, err
	}

	log.Debug(gctx, "Authority details successfully queried")

	authorityDetails := make(map[string]domain.AuthorityDetails)

	// Fetch authority details from post_management_master table
	fields := []struct {
		fieldName string
		postID    int32
	}{

		{"GDS Leave Sanctioning Authority - Subdvn Head", authority.GDSLeaveSancAuthority1},
		{"GDS Leave Sanctioning Authority - SSP", authority.GDSLeaveSancAuthority2},
		{"Reporting Authority", authority.ReportingAuthority},
		{"APAR Reporting Authority", authority.AparReportingAuthority},
		{"APAR Review Authority", authority.AparReviewAuthority},
		{"APAR Accepting Authority", authority.AparAcceptingAuthority},
		{"APAR Represent Authority", authority.AparRepresentAuthority},
		{"Service Book Authority - Admin", authority.ServiceBookApproveAuthority1},
		{"CL Sanctioning Authority", authority.LeaveSancAuthority1},
		{"EL and other Leave Sanctioning Authority", authority.LeaveSancAuthority2},
		{"Special Leave Sanctioning Authority (study leave etc)", authority.LeaveSancAuthority3},
		{"Pay Approve Authority - DDO level", authority.PayApproveAuthority1},
		//{"Pay Approve Authority - Admin level", authority.PayApproveAuthority2},
		{"CL Leave Forwarding - maker", authority.LeaveFWDAuthority1},
		{"EL Leave Forwarding - maker", authority.LeaveFWDAuthority2},
		{"Pay Forwarding Authority - DDO level maker", authority.PayFWDAuthority1},
		{"Pay Forwarding Authority - Admin level maker", authority.PayFWDAuthority2},
		{"Appointing Authority", authority.AppointingAuthority},
		{"Disciplinary Authority", authority.DisciplinaryAuthority},
		{"DDO Authority", authority.DdoAuthority},
		{"Admin Authority", authority.AdminAuthority},
		{"Pension Sanctioning Authority", authority.PensionSanctioningAuthority},
		{"Pension Authorising Authority", authority.PensionAuthorisingAuthority},
		{"Service Book Authority - DDO", authority.ServiceBookApproveAuthority2},
		{"Role Authority", authority.RoleAuthority},
		{"Service Book Forwarding - Admin", authority.ServiceBookForwardAuthority1},
		{"Service Book Forwarding - DDO", authority.ServiceBookForwardAuthority2},
		{"Vigilance Forwarding Authority", authority.VigilenceMakerAuthority},
		{AdminOffice, authority.AdminOffice},
	}

	for _, field := range fields {
		if field.fieldName == AdminOffice {
			officeName, err := ear.getAdminOfficeName(ctx, field.postID)
			if err != nil {
				log.Debug(gctx, "Error fetching admin office name:", err)
				continue
			}

			authorityDetails[field.fieldName] = domain.AuthorityDetails{
				AuthorityPost:        int(field.postID),
				DesignationName:      officeName,
				OfficeID:             int(field.postID),
				OfficeName:           officeName,
				AuthorityDescription: field.fieldName,
				AuthorityName:        field.fieldName,
			}
		} else {
			designationName, officeID, officeName, err := ear.getDesignationAndOffice(ctx, field.postID)
			if err != nil {
				log.Error(ctx, DesandSourceOffice, field.fieldName, err)
				continue // Continue to the next authority detail
			}

			log.Debug(gctx, "Designation and office details successfully fetched for field %s\n", field.fieldName)

			authorityDetails[field.fieldName] = domain.AuthorityDetails{
				AuthorityName:   field.fieldName,
				AuthorityPost:   int(field.postID),
				DesignationName: designationName,
				OfficeID:        officeID,
				OfficeName:      officeName,
			}
		}
	}

	log.Debug(gctx, "Authority details successfully processed")

	return authorityDetails, nil
}

// Helper function to fetch designation and office details
func (pmr *PosttoPostMappingRepository) getDesignationAndOffice(ctx context.Context, postID int32) (string, int, string, error) {
	var designationName string
	var officeID int
	var officeName string

	err := pmr.db.QueryRow(ctx, `
		SELECT designation, office_id, office_name 
		FROM pmdm.post_management_master 
		WHERE post_id = $1
	`, postID).Scan(&designationName, &officeID, &officeName)
	if err != nil {
		return "", 0, "", err
	}

	return designationName, officeID, officeName, nil
}

func (ear *PosttoPostMappingRepository) GetAuthorityDetailsForMultiplePostID(gctx *gin.Context, postIDs []int) (map[int32]map[string]domain.AuthorityDetails, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), ear.cfg.GetDuration(DBtimeout))
	defer cancel()

	// Prepare the query with a variable number of placeholders
	query := `
		SELECT 
			employee_post_id,
			COALESCE(gds_leave_sanc_authority_1, 0), 
			COALESCE(gds_leave_sanc_authority_2, 0),
			COALESCE(reporting_authority, 0),
			COALESCE(apar_reporting_authority, 0),
			COALESCE(apar_review_authority, 0),
			COALESCE(apar_controlling_authority, 0),
			COALESCE(apar_appellate_authority, 0),
			COALESCE(service_book_approve_authority1, 0),
			COALESCE(leave_sanc_authority_1, 0),
			COALESCE(leave_sanc_authority_2, 0),
			COALESCE(leave_sanc_authority_3, 0),
			COALESCE(pay_approve_authority1, 0),
			
			COALESCE(leave_fwd_authority1, 0),
			COALESCE(leave_fwd_authority2, 0),
			COALESCE(pay_fwd_authority1, 0),
			COALESCE(pay_fwd_authority2, 0),
			COALESCE(appointing_authority, 0),
			COALESCE(disciplinary_authority, 0),
			COALESCE(ddo_authority, 0),
			COALESCE(admin_authority, 0),
			COALESCE(pension_sanctioning_authority, 0),
			COALESCE(pension_authorising_authority, 0),
			COALESCE(service_book_approve_authority2, 0),
			COALESCE(role_authority, 0),
			COALESCE(service_book_foward_authority1, 0),
			COALESCE(service_book_foward_authority2, 0),
			COALESCE(apar_custodian_authority,0)
		FROM pmdm.post_mapping_detail 
		WHERE employee_post_id = ANY($1)
	`

	// Execute the query
	rows, err := ear.db.Query(ctx, query, postIDs)
	if err != nil {
		log.Debug(gctx, ErrorQueryAuthority, err)
		return nil, err
	}
	defer rows.Close()

	authorityDetailsMap := make(map[int32]map[string]domain.AuthorityDetails)

	for rows.Next() {
		var postID int32
		var authority domain.PosttoPostMap

		err := rows.Scan(
			&postID,
			&authority.GDSLeaveSancAuthority1,
			&authority.GDSLeaveSancAuthority2,
			&authority.ReportingAuthority,
			&authority.AparReportingAuthority,
			&authority.AparReviewAuthority,
			&authority.AparAcceptingAuthority,
			&authority.AparRepresentAuthority,
			&authority.ServiceBookApproveAuthority1,
			&authority.LeaveSancAuthority1,
			&authority.LeaveSancAuthority2,
			&authority.LeaveSancAuthority3,
			&authority.PayApproveAuthority1,
			//&authority.PayApproveAuthority2,
			&authority.LeaveFWDAuthority1,
			&authority.LeaveFWDAuthority2,
			&authority.PayFWDAuthority1,
			&authority.PayFWDAuthority2,
			&authority.AppointingAuthority,
			&authority.DisciplinaryAuthority,
			&authority.DdoAuthority,
			&authority.AdminAuthority,
			&authority.PensionSanctioningAuthority,
			&authority.PensionAuthorisingAuthority,
			&authority.ServiceBookApproveAuthority2,
			&authority.RoleAuthority,
			&authority.ServiceBookForwardAuthority1,
			&authority.ServiceBookForwardAuthority2,
		)
		if err != nil {
			log.Debug(gctx, "Error scanning authority details:", err)
			return nil, err
		}

		log.Debug(gctx, "Authority details successfully queried for post ID", postID)

		authorityDetails := make(map[string]domain.AuthorityDetails)

		fields := []struct {
			fieldName string
			postID    int32
		}{
			{"GDSLeaveSancAuthority1", authority.GDSLeaveSancAuthority1},
			{"GDSLeaveSancAuthority2", authority.GDSLeaveSancAuthority2},
			{"ReportingAuthority", authority.ReportingAuthority},
			{"AparReportingAuthority", authority.AparReportingAuthority},
			{"AparReviewAuthority", authority.AparReviewAuthority},
			{"AparAcceptingAuthority", authority.AparAcceptingAuthority},
			{"AparRepresentAuthority", authority.AparRepresentAuthority},
			{"ServiceBookApproveAuthority1", authority.ServiceBookApproveAuthority1},
			{"LeaveSancAuthority1", authority.LeaveSancAuthority1},
			{"LeaveSancAuthority2", authority.LeaveSancAuthority2},
			{"LeaveSancAuthority3", authority.LeaveSancAuthority3},
			{"PayApproveAuthority1", authority.PayApproveAuthority1},
			//{"PayApproveAuthority2", authority.PayApproveAuthority2},
			{"LeaveFWDAuthority1", authority.LeaveFWDAuthority1},
			{"LeaveFWDAuthority2", authority.LeaveFWDAuthority2},
			{"PayFWDAuthority1", authority.PayFWDAuthority1},
			{"PayFWDAuthority2", authority.PayFWDAuthority2},
			{"AppointingAuthority", authority.AppointingAuthority},
			{"DisciplinaryAuthority", authority.DisciplinaryAuthority},
			{"DdoAuthority", authority.DdoAuthority},
			{"AdminAuthority", authority.AdminAuthority},
			{"PensionSanctioningAuthority", authority.PensionSanctioningAuthority},
			{"PensionAuthorisingAuthority", authority.PensionAuthorisingAuthority},
			{"ServiceBookApproveAuthority2", authority.ServiceBookApproveAuthority2},
			{"RoleAuthority", authority.RoleAuthority},
			{"ServiceBookFowardAuthority1", authority.ServiceBookForwardAuthority1},
			{"ServiceBookFowardAuthority2", authority.ServiceBookForwardAuthority2},
		}

		for _, field := range fields {
			designationName, officeID, officeName, err := ear.getDesignationAndOffice(ctx, field.postID)
			if err != nil {
				log.Debug(gctx, DesandSourceOffice, field.fieldName, err)
				continue
			}

			log.Debug(gctx, "Designation and office details successfully fetched for field %s\n", field.fieldName)

			authorityDetails[field.fieldName] = domain.AuthorityDetails{
				AuthorityName:   field.fieldName,
				AuthorityPost:   int(field.postID),
				DesignationName: designationName,
				OfficeID:        officeID,
				OfficeName:      officeName,
			}
		}

		authorityDetailsMap[postID] = authorityDetails
	}

	if err := rows.Err(); err != nil {
		log.Debug(gctx, "Error iterating through rows:", err)
		return nil, err
	}

	log.Debug(gctx, "Authority details successfully processed for all post IDs")

	return authorityDetailsMap, nil
}

func isValidField(fieldName string) bool {
	validFields := map[string]bool{
		"gds_leave_sanc_authority_1":      true,
		"gds_leave_sanc_authority_2":      true,
		"reporting_authority":             true,
		"apar_reporting_authority":        true,
		"apar_review_authority":           true,
		"apar_controlling_authority":      true,
		"apar_appellate_authority":        true,
		"service_book_approve_authority1": true,
		"leave_sanc_authority_1":          true,
		"leave_sanc_authority_2":          true,
		"leave_sanc_authority_3":          true,
		"pay_approve_authority1":          true,
		//"pay_approve_authority2":          true,
		"leave_fwd_authority1":            true,
		"leave_fwd_authority2":            true,
		"pay_fwd_authority1":              true,
		"pay_fwd_authority2":              true,
		"appointing_authority":            true,
		"disciplinary_authority":          true,
		"ddo_authority":                   true,
		"admin_authority":                 true,
		"pension_sanctioning_authority":   true,
		"pension_authorising_authority":   true,
		"service_book_approve_authority2": true,
		"role_authority":                  true,
		"service_book_foward_authority1":  true,
		"service_book_foward_authority2":  true,
		"vigilence_maker_authority":       true,
		"admin_office":                    true,
		"apar_custodian_authority":        true,
		// Add other valid field names here
	}
	return validFields[fieldName]
}

// func buildUpdateMap(fieldNames []string, fieldValues []interface{}) map[string]interface{} {
// 	updateMap := make(map[string]interface{})
// 	for i := range fieldNames {
// 		updateMap[fieldNames[i]] = fieldValues[i]
// 	}
// 	return updateMap
// }

func buildUpdateMap(fieldNames []string, fieldValues []interface{}) map[string]interface{} {
	updateMap := make(map[string]interface{})

	if len(fieldValues) == 1 {
		for _, fieldName := range fieldNames {
			updateMap[fieldName] = fieldValues[0]
		}
	} else {
		for i := range fieldNames {
			if i < len(fieldValues) {
				updateMap[fieldNames[i]] = fieldValues[i]
			} else {
				updateMap[fieldNames[i]] = nil // or use a default value like 0 or ""
			}
		}
	}

	return updateMap
}

func (pmr *PosttoPostMappingRepository) CreatePostMappingDetailMaker(gctx *gin.Context, empPostIDs []int, fieldNames []string, fieldValues []interface{}, officeID int, approvePostID string, createdBy string) ([]domain.PosttoPostMap, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	log.Debug(gctx, "Starting CreatePostMappingDetailMaker")

	// Validate the field names
	for _, fieldName := range fieldNames {
		if !isValidField(fieldName) {
			log.Debug(gctx, "Invalid field name:", fieldName)
			return nil, errors.New("invalid field name")
		}
	}
	log.Debug(gctx, "Field names validated")

	var updatedResponses []domain.PosttoPostMap

	// Iterate over each EmpPostID
	for _, empPostID := range empPostIDs {
		// Check if there's already a pending record for the current EmpPostID
		var approveStatus string
		checkQuery := `SELECT approve_status FROM pmdm.post_mapping_detail_maker WHERE employee_post_id = $1 AND approve_status = 'Pending'`
		err := pmr.db.QueryRow(ctx, checkQuery, empPostID).Scan(&approveStatus)
		if err != nil && err != pgx.ErrNoRows {
			log.Debug(gctx, "Error checking pending status for empPostID:", empPostID, ":", err)
			return nil, err
		}

		if approveStatus == "Pending" {
			// Update the existing record with new values
			updateMap := buildUpdateMap(fieldNames, fieldValues)
			updateQuery := `UPDATE pmdm.post_mapping_detail_maker SET `

			setClauses := make([]string, 0)
			values := make([]interface{}, 0)
			for i, fieldName := range fieldNames {
				setClauses = append(setClauses, fmt.Sprintf("%s = $%d", fieldName, i+1))
				values = append(values, updateMap[fieldName])
			}
			setClauses = append(setClauses, fmt.Sprintf("approve_post_id = $%d", len(values)+1))
			setClause := strings.Join(setClauses, ", ")

			updateQuery += setClause + ` WHERE employee_post_id = $` + fmt.Sprintf("%d", len(values)+2) + ` AND approve_status = 'Pending'`
			values = append(values, approvePostID, empPostID)

			_, err = pmr.db.Exec(ctx, updateQuery, values...)
			if err != nil {
				log.Debug(gctx, "Error updating record for empPostID:", empPostID, ":", err)
				return nil, err
			}

			// // Log the update
			// err = pmr.logUpdate(ctx, empPostID, fieldNames, fieldValues)
			// if err != nil {
			// 	return nil, err
			// }
		} else {
			// Insert the new record with approve_status set to 'Pending'
			updateMap := buildUpdateMap(fieldNames, fieldValues)

			// Construct the values slice for insertion
			values := []interface{}{empPostID, officeID, "Pending", time.Now(), createdBy, approvePostID} // Include created_date, created_by, and approve_post_id

			// Add the field values from the updateMap to the values slice
			for _, fieldName := range fieldNames {
				values = append(values, updateMap[fieldName])
			}

			// Insert the new record
			insertBuilder := dblib.Psql.Insert("pmdm.post_mapping_detail_maker").
				Columns(append([]string{"employee_post_id", "employee_office_id", "approve_status", "created_date", "created_by", "approve_post_id"}, fieldNames...)...).
				Values(values...).
				Suffix("RETURNING employee_post_id, approve_status, created_date")

			sql, args, err := insertBuilder.ToSql()
			if err != nil {
				log.Debug(gctx, "Error building insert SQL query for empPostID:", empPostID, ":", err)
				return nil, err
			}
			log.Debug(gctx, "Insert SQL query built successfully for empPostID:", empPostID)

			// Execute the insert query for the current EmpPostID
			rows, err := pmr.db.Query(ctx, sql, args...)
			if err != nil {
				log.Debug(ctx, "Error executing insert SQL query for empPostID:", empPostID, ":", err)
				return nil, err
			}
			defer rows.Close()

			// Scan and map the inserted response for the current EmpPostID
			for rows.Next() {
				var updatedResponse domain.PosttoPostMap
				var createdDate time.Time
				if err := rows.Scan(&updatedResponse.EmployeePostID, &approveStatus, &createdDate); err != nil {
					log.Debug(ctx, "Error scanning row for empPostID:", empPostID, ":", err)
					return nil, err
				}

				// Set the CreatedDate
				// updatedResponse.UpdatedDate = createdDate

				// Append each field updated to the response
				for j, fieldName := range fieldNames {
					var newValue interface{}
					if j < len(fieldValues) {
						newValue = fieldValues[j]
					} else {
						newValue = nil
					}
					newResponse := domain.PosttoPostMap{
						EmployeePostID: empPostID,
						UpdatedDate:    time.Now(),
						FieldUpdated:   fieldName,
						NewValue:       newValue,
					}
					updatedResponses = append(updatedResponses, newResponse)
				}

				// Log the insertion
				err = pmr.logUpdate(ctx, empPostID, fieldNames, fieldValues)
				if err != nil {
					return nil, err
				}
			}

			if err := rows.Err(); err != nil {
				log.Debug(ctx, "Error in result set for empPostID:", empPostID, ":", err)
				return nil, err
			}
			log.Debug(ctx, "Rows scanned and mapped successfully for empPostID:", empPostID)
		}
	}

	log.Debug(gctx, "CreatePostMappingDetailMaker completed successfully")
	return updatedResponses, nil
}

func (pmr *PosttoPostMappingRepository) logUpdate(ctx context.Context, empPostID int, fieldNames []string, fieldValues []interface{}) error {
	// Create a map for the field values
	updateMap := buildUpdateMap(fieldNames, fieldValues)

	// Construct the columns and placeholders dynamically
	columns := []string{"employee_post_id"}
	placeholders := []string{"$1"}
	values := []interface{}{empPostID}

	// Add dynamic fields
	for i, fieldName := range fieldNames {
		columns = append(columns, fieldName)
		placeholders = append(placeholders, fmt.Sprintf("$%d", i+2)) // Adjusting index for placeholders
		values = append(values, updateMap[fieldName])
	}

	// Construct the query for logging
	logQuery := fmt.Sprintf(
		`INSERT INTO pmdm.post_mapping_detail_maker_log (%s) VALUES (%s)`,
		strings.Join(columns, ", "),
		strings.Join(placeholders, ", "),
	)

	// Execute the insert query
	_, err := pmr.db.Exec(ctx, logQuery, values...)
	if err != nil {
		log.Debug(ctx, "Error logging update for empPostID:", empPostID, ":", err)
		return err
	}

	return nil
}

func (ear *PosttoPostMappingRepository) GetPostMappingMasterMaker(gctx *gin.Context, postID int32) (map[string][]domain.AuthorityDetails, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), ear.cfg.GetDuration(DBtimeout))
	defer cancel()

	// Query the database to fetch all authority details for the given employee_post_id
	rows, err := ear.db.Query(ctx, `
        SELECT
            COALESCE(gds_leave_sanc_authority_1, 0),
            COALESCE(gds_leave_sanc_authority_2, 0),
            COALESCE(reporting_authority, 0),
            COALESCE(apar_reporting_authority, 0),
            COALESCE(apar_review_authority, 0),
            COALESCE(apar_controlling_authority, 0),
            COALESCE(apar_appellate_authority, 0),
            COALESCE(service_book_approve_authority1, 0),
            COALESCE(leave_sanc_authority_1, 0),
            COALESCE(leave_sanc_authority_2, 0),
            COALESCE(leave_sanc_authority_3, 0),
            COALESCE(pay_approve_authority1, 0),
            COALESCE(leave_fwd_authority1, 0),
            COALESCE(leave_fwd_authority2, 0),
            COALESCE(pay_fwd_authority1, 0),
            COALESCE(pay_fwd_authority2, 0),
            COALESCE(appointing_authority, 0),
            COALESCE(disciplinary_authority, 0),
            COALESCE(ddo_authority, 0),
            COALESCE(admin_authority, 0),
            COALESCE(pension_sanctioning_authority, 0),
            COALESCE(pension_authorising_authority, 0),
            COALESCE(service_book_approve_authority2, 0),
            COALESCE(role_authority, 0),
            COALESCE(service_book_foward_authority1, 0),
            COALESCE(service_book_foward_authority2, 0),
			COALESCE(vigilence_maker_authority, 0),
			COALESCE(admin_office, 0),
			COALESCE(apar_custodian_authority,0)

        FROM pmdm.post_mapping_detail_maker WHERE employee_post_id = $1 AND approve_status = 'Pending'
    `, postID)
	if err != nil {
		log.Error(gctx, ErrorQueryAuthority, err)
		return nil, err
	}
	defer rows.Close()

	log.Error(gctx, "Authority details successfully queried")

	authorityDetails := make(map[string][]domain.AuthorityDetails)

	fields := []struct {
		fieldName  string
		columnName string
	}{
		{"GDS Leave Sanctioning Authority level 1", "gds_leave_sanc_authority_1"},
		{"GDS Leave Sanctioning Authority level 2", "gds_leave_sanc_authority_2"},
		{"Reporting Authority", "reporting_authority"},
		{"Apar Reporting Authority", "apar_reporting_authority"},
		{"Apar Review Authority", "apar_review_authority"},
		{"Apar Accepting Authority", "apar_controlling_authority"},
		{"Apar Represent Authority", "apar_appellate_authority"},
		{"Service Book Authority - Admin", "service_book_approve_authority1"},
		{"CL Sanctioning Authority", "leave_sanc_authority_1"},
		{"EL and other Leave Sanctioning Authority", "leave_sanc_authority_2"},
		{"Special Leave Sanctioning Authority", "leave_sanc_authority_3"},
		{"Pay Approve Authority - DDO", "pay_approve_authority1"},
		//{"Pay Approve Authority  - Admin", "pay_approve_authority2"},
		{"CL Leave Forwarding ", "leave_fwd_authority1"},
		{"EL Leave Forwarding ", "leave_fwd_authority2"},
		{"Pay Forwarding Authority DDO", "pay_fwd_authority1"},
		{"Pay Forwarding Authority -  Admin", "pay_fwd_authority2"},
		{"Appointing Authority", "appointing_authority"},
		{"Disciplinary Authority", "disciplinary_authority"},
		{"DDO Authority", "ddo_authority"},
		{"Admin Authority", "admin_authority"},
		{"Pension Sanctioning Authority", "pension_sanctioning_authority"},
		{"Pension Approving Authority", "pension_authorising_authority"},
		{"Service Book Authority - DDO", "service_book_approve_authority2"},
		{"Role Authority", "role_authority"},
		{"Service Book Forwading - Admin", "service_book_foward_authority1"},
		{"Service Book Forwarding  - DDO", "service_book_foward_authority2"},
		{"Vigilance Forwarding Authority", "vigilence_maker_authority"},
		{AdminOffice, "admin_office"},
		{"Apar Custodian Auhtority", "apar_custodian_authority"},
	}

	for rows.Next() {
		var authority domain.PosttoPostMap
		err := rows.Scan(
			&authority.GDSLeaveSancAuthority1,
			&authority.GDSLeaveSancAuthority2,
			&authority.ReportingAuthority,
			&authority.AparReportingAuthority,
			&authority.AparReviewAuthority,
			&authority.AparAcceptingAuthority,
			&authority.AparRepresentAuthority,
			&authority.ServiceBookApproveAuthority1,
			&authority.LeaveSancAuthority1,
			&authority.LeaveSancAuthority2,
			&authority.LeaveSancAuthority3,
			&authority.PayApproveAuthority1,
			//&authority.PayApproveAuthority2,
			&authority.LeaveFWDAuthority1,
			&authority.LeaveFWDAuthority2,
			&authority.PayFWDAuthority1,
			&authority.PayFWDAuthority2,
			&authority.AppointingAuthority,
			&authority.DisciplinaryAuthority,
			&authority.DdoAuthority,
			&authority.AdminAuthority,
			&authority.PensionSanctioningAuthority,
			&authority.PensionAuthorisingAuthority,
			&authority.ServiceBookApproveAuthority2,
			&authority.RoleAuthority,
			&authority.ServiceBookForwardAuthority1,
			&authority.ServiceBookForwardAuthority2,
			&authority.VigilenceMakerAuthority,
			&authority.AdminOffice,
			&authority.AparCustodianAuhtority,
		)
		if err != nil {
			log.Error(gctx, "Error scanning authority details:", err)
			continue
		}

		for _, field := range fields {
			var postID int32
			switch field.columnName {
			case "gds_leave_sanc_authority_1":
				postID = authority.GDSLeaveSancAuthority1
			case "gds_leave_sanc_authority_2":
				postID = authority.GDSLeaveSancAuthority2
			case "reporting_authority":
				postID = authority.ReportingAuthority
			case "apar_reporting_authority":
				postID = authority.AparReportingAuthority
			case "apar_review_authority":
				postID = authority.AparReviewAuthority
			case "apar_controlling_authority":
				postID = authority.AparAcceptingAuthority
			case "apar_appellate_authority":
				postID = authority.AparRepresentAuthority
			case "service_book_approve_authority1":
				postID = authority.ServiceBookApproveAuthority1
			case "leave_sanc_authority_1":
				postID = authority.LeaveSancAuthority1
			case "leave_sanc_authority_2":
				postID = authority.LeaveSancAuthority2
			case "leave_sanc_authority_3":
				postID = authority.LeaveSancAuthority3
			case "pay_approve_authority1":
				postID = authority.PayApproveAuthority1
			// case "pay_approve_authority2":
			// 	postID = authority.PayApproveAuthority2
			case "leave_fwd_authority1":
				postID = authority.LeaveFWDAuthority1
			case "leave_fwd_authority2":
				postID = authority.LeaveFWDAuthority2
			case "pay_fwd_authority1":
				postID = authority.PayFWDAuthority1
			case "pay_fwd_authority2":
				postID = authority.PayFWDAuthority2
			case "appointing_authority":
				postID = authority.AppointingAuthority
			case "disciplinary_authority":
				postID = authority.DisciplinaryAuthority
			case "ddo_authority":
				postID = authority.DdoAuthority
			case "admin_authority":
				postID = authority.AdminAuthority
			case "pension_sanctioning_authority":
				postID = authority.PensionSanctioningAuthority
			case "pension_authorising_authority":
				postID = authority.PensionAuthorisingAuthority
			case "service_book_approve_authority2":
				postID = authority.ServiceBookApproveAuthority2
			case "role_authority":
				postID = authority.RoleAuthority
			case "service_book_foward_authority1":
				postID = authority.ServiceBookForwardAuthority1
			case "service_book_foward_authority2":
				postID = authority.ServiceBookForwardAuthority2
			case "vigilence_maker_authority":
				postID = authority.VigilenceMakerAuthority
			case "admin_office":
				postID = authority.AdminOffice
			case "apar_custodian_authority":
				postID = authority.AparCustodianAuhtority
			}

			if field.fieldName == AdminOffice {
				officeName, err := ear.getAdminOfficeName(ctx, postID)
				if err != nil {
					log.Error(gctx, "Error fetching admin office name:", err)
					continue
				}

				adminOfficeDetails := domain.AuthorityDetails{
					AuthorityPost:        int(postID),
					DesignationName:      officeName,
					OfficeID:             int(postID),
					OfficeName:           officeName,
					AuthorityDescription: field.fieldName,
					AuthorityName:        field.columnName,
				}

				authorityDetails[field.fieldName] = append(authorityDetails[field.fieldName], adminOfficeDetails)

			}

			designationName, officeID, officeName, err := ear.getDesignationAndOffice(ctx, postID)

			if err != nil {
				log.Error(gctx, DesandSourceOffice, field.fieldName, err)
				continue
			}

			authorityDetails[field.fieldName] = append(authorityDetails[field.fieldName], domain.AuthorityDetails{
				AuthorityPost:        int(postID),
				DesignationName:      designationName,
				OfficeID:             officeID,
				OfficeName:           officeName,
				AuthorityDescription: field.fieldName,
				AuthorityName:        field.columnName,
			})
		}
	}

	log.Error(gctx, "Successfully populated authority details map")

	return authorityDetails, nil
}
func (ear *PosttoPostMappingRepository) getAdminOfficeName(ctx context.Context, officeID int32) (string, error) {
	var officeName string

	err := ear.db.QueryRow(ctx, `
        SELECT office_name FROM pmdm.kafka_office_master WHERE office_id = $1
    `, officeID).Scan(&officeName)

	if err != nil {
		log.Error(ctx, "Error querying admin office name:", err)
		return "", err // Return an empty string and error if there's an error
	}

	return officeName, nil
}
func (pmr *PosttoPostMappingRepository) ApprovePostMappingDetailMaker(gctx *gin.Context, details []domain.ApprovePostMappingDetail) (string, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	// Start a transaction
	tx, err := pmr.db.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return "", err
	}
	defer tx.Rollback(ctx)

	// Create a map to aggregate field updates by employee_post_id
	fieldUpdates := make(map[int][]domain.ApprovePostMappingDetail)

	for _, detail := range details {
		fieldUpdates[detail.EmployeePostID] = append(fieldUpdates[detail.EmployeePostID], detail)
	}

	// Update the maker table
	for empPostID, fields := range fieldUpdates {
		updateQuery := `UPDATE pmdm.post_mapping_detail_maker SET approve_status = 'Approved', approved_date = $1, approved_by = $2, remarks = $3, updated_date = $4 , updated_by = $5`
		updateArgs := []interface{}{time.Now(), fields[0].ApprovedBy, fields[0].Remarks, time.Now(), fields[0].ApprovedBy}

		for i, field := range fields {
			if field.FieldName != "" {
				updateQuery += `, ` + field.FieldName + ` = $` + strconv.Itoa(i+6)
				updateArgs = append(updateArgs, field.FieldValue)
			}
		}

		updateQuery += ` WHERE employee_post_id = $` + strconv.Itoa(len(updateArgs)+1) + ` AND approve_status = 'Pending'`
		updateArgs = append(updateArgs, empPostID)

		cmdTag, err := tx.Exec(ctx, updateQuery, updateArgs...)
		if err != nil {
			return "", err
		}
		if cmdTag.RowsAffected() == 0 {
			return "", errors.New("no pending record found to approve for employee_post_id: " + strconv.Itoa(empPostID))
		}
	}

	// Insert or update in the master table
	for _, field := range details {
		insertQuery := `
	INSERT INTO pmdm.post_mapping_detail (
		employee_post_id, employee_office_id, ` + field.FieldName + `, updated_date, updated_by
	) VALUES (
		$1, $2, $3, $4, $5
	)
	ON CONFLICT (employee_post_id) DO UPDATE SET
		employee_office_id = EXCLUDED.employee_office_id,
		` + field.FieldName + ` = EXCLUDED.` + field.FieldName + `,
		updated_date = EXCLUDED.updated_date,
		updated_by = EXCLUDED.updated_by
	RETURNING employee_post_id, employee_office_id
`

		var fieldValue int32
		if field.Status == "Rejected" {
			fieldValue = 0
		} else {
			fieldValue = field.FieldValue
		}

		row := tx.QueryRow(ctx, insertQuery,
			field.EmployeePostID, field.OfficeID, fieldValue, time.Now(), field.ApprovedBy,
		)

		var masterRecord domain.PosttoPostMap
		if err := row.Scan(
			&masterRecord.EmployeePostID,
			&masterRecord.OfficeID,
		); err != nil {
			return "", err
		}
	}

	// Commit the transaction
	if err := tx.Commit(ctx); err != nil {
		return "", err
	}

	return "record insert and update successfully", nil
}

func (ear *PosttoPostMappingRepository) GetMasterAuthoritiesDeatils(gctx *gin.Context, postid int) ([]domain.MasterAuthority, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), ear.cfg.GetDuration(DBtimeout))
	defer cancel()
	var authorities []domain.MasterAuthority

	// Fetching authority details based on post_id
	query := dblib.Psql.Select("cadre_name", "designation", "role_mapping_id", "authority_description").
		From("pmdm.master_authorities").
		Where(sq.Eq{"post_id": postid})

	sql, args, err := query.ToSql()
	if err != nil {
		log.Debug(gctx, "error at execute", err)
		return nil, err
	}

	rows, err := ear.db.Query(ctx, sql, args...)
	if err != nil {
		log.Debug(gctx, ErrorFor, err)
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var authority domain.MasterAuthority
		err := rows.Scan(&authority.CadreName, &authority.Designation, &authority.RoleMappingID, &authority.AuthorityDescription)
		if err != nil {
			log.Debug(gctx, ErrorFor, err)
			return nil, err
		}
		authorities = append(authorities, authority)
	}
	if err := rows.Err(); err != nil {
		log.Debug(gctx, ErrorFor, err)
		return nil, err
	}

	return authorities, nil
}

func (pmr *PosttoPostMappingRepository) GetPostMappingMakerDetails(gctx *gin.Context, approvePostID string, reqMetadata port.MetaDataRequest) ([]domain.PosttoPostMap, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := `
        SELECT 
            COALESCE(pmdm.employee_post_id, 0), 
            COALESCE(pmdm.gds_leave_sanc_authority_1, 0), 
            COALESCE(pmdm.gds_leave_sanc_authority_2, 0), 
            COALESCE(pmdm.reporting_authority, 0), 
            COALESCE(pmdm.apar_reporting_authority, 0), 
            COALESCE(pmdm.apar_review_authority, 0), 
            COALESCE(pmdm.apar_controlling_authority, 0), 
            COALESCE(pmdm.apar_appellate_authority, 0), 
            COALESCE(pmdm.service_book_approve_authority1, 0), 
            COALESCE(pmdm.leave_sanc_authority_1, 0), 
            COALESCE(pmdm.leave_sanc_authority_2, 0), 
            COALESCE(pmdm.leave_sanc_authority_3, 0), 
            COALESCE(pmdm.pay_approve_authority1, 0), 
            
            COALESCE(pmdm.leave_fwd_authority1, 0), 
            COALESCE(pmdm.leave_fwd_authority2, 0), 
            COALESCE(pmdm.pay_fwd_authority1, 0), 
            COALESCE(pmdm.pay_fwd_authority2, 0), 
            COALESCE(pmdm.appointing_authority, 0), 
            COALESCE(pmdm.disciplinary_authority, 0), 
            COALESCE(pmdm.ddo_authority, 0), 
            COALESCE(pmdm.admin_authority, 0), 
            COALESCE(pmdm.pension_sanctioning_authority, 0), 
            COALESCE(pmdm.pension_authorising_authority, 0), 
            COALESCE(pmdm.service_book_approve_authority2, 0), 
            COALESCE(pmdm.role_authority, 0), 
            COALESCE(pmdm.service_book_foward_authority1, 0), 
            COALESCE(pmdm.service_book_foward_authority2, 0), 
            COALESCE(pmdm.post_map_id, 0), 
            COALESCE(pmdm.updated_date, '1970-01-01'::timestamp), 
            COALESCE(pmdm.updated_by, ''), 
            COALESCE(pmdm.vigilence_maker_authority, 0), 
            COALESCE(pmdm.admin_office, 0),
            COALESCE(pmdm.approve_status, ''),
            COALESCE(pmdm.employee_office_id, 0),
            COALESCE(kom.office_name, ''),
            COALESCE(pm.post_name, ''),
			COALESCE(pmdm.created_date, '1970-01-01'::timestamp), 
            COALESCE(pmdm.created_by, '')
        FROM 
            pmdm.post_mapping_detail_maker pmdm
        LEFT JOIN 
            kafka_office_master kom ON pmdm.employee_office_id = kom.office_id
        LEFT JOIN 
            post_management_master pm ON pmdm.employee_post_id = pm.post_id
        WHERE 
            pmdm.approve_post_id = $1 AND pmdm.approve_status = 'Pending'`

	rows, err := pmr.db.Query(ctx, query, approvePostID)
	if err != nil {
		log.Error(gctx, "Database query error: %v", err)
		return nil, fmt.Errorf("error querying post mapping maker details: %w", err)
	}
	defer rows.Close()

	var postMappings []domain.PosttoPostMap

	for rows.Next() {
		var postMapping domain.PosttoPostMap

		err := rows.Scan(
			&postMapping.EmployeePostID,
			&postMapping.GDSLeaveSancAuthority1,
			&postMapping.GDSLeaveSancAuthority2,
			&postMapping.ReportingAuthority,
			&postMapping.AparReportingAuthority,
			&postMapping.AparReviewAuthority,
			&postMapping.AparAcceptingAuthority,
			&postMapping.AparRepresentAuthority,
			&postMapping.ServiceBookApproveAuthority1,
			&postMapping.LeaveSancAuthority1,
			&postMapping.LeaveSancAuthority2,
			&postMapping.LeaveSancAuthority3,
			&postMapping.PayApproveAuthority1,
			//&postMapping.PayApproveAuthority2,
			&postMapping.LeaveFWDAuthority1,
			&postMapping.LeaveFWDAuthority2,
			&postMapping.PayFWDAuthority1,
			&postMapping.PayFWDAuthority2,
			&postMapping.AppointingAuthority,
			&postMapping.DisciplinaryAuthority,
			&postMapping.DdoAuthority,
			&postMapping.AdminAuthority,
			&postMapping.PensionSanctioningAuthority,
			&postMapping.PensionAuthorisingAuthority,
			&postMapping.ServiceBookApproveAuthority2,
			&postMapping.RoleAuthority,
			&postMapping.ServiceBookForwardAuthority1,
			&postMapping.ServiceBookForwardAuthority2,
			&postMapping.PostMapID,
			&postMapping.UpdatedDate,
			&postMapping.UpdatedBy,
			&postMapping.VigilenceMakerAuthority,
			&postMapping.AdminOffice,
			&postMapping.ApproveStatus,
			&postMapping.OfficeID,
			&postMapping.OfficeName,
			&postMapping.PostName,
			&postMapping.CreatedDate,
			&postMapping.CreatedBy,
		)
		if err != nil {
			log.Error(gctx, "Error scanning row: %v", err)
			return nil, fmt.Errorf("error scanning post mapping details: %w", err)
		}

		postMappings = append(postMappings, postMapping)
	}

	if err := rows.Err(); err != nil {
		log.Error(gctx, "Rows iteration error: %v", err)
		return nil, fmt.Errorf("error iterating over rows: %w", err)
	}

	return postMappings, nil
}
func generatePlaceholders(n int) []string {
	placeholders := make([]string, n)
	for i := 1; i <= n; i++ {
		placeholders[i-1] = fmt.Sprintf("$%d", i)
	}
	return placeholders
}
func (pmr *PosttoPostMappingRepository) ApprovePostMappingDetailMaker2(gctx *gin.Context, details []domain.ApprovePostMappingDetail) (string, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	// Start a transaction
	tx, err := pmr.db.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return "", err
	}
	defer tx.Rollback(ctx)

	// Group details by EmployeePostID
	fieldUpdates := make(map[int][]domain.ApprovePostMappingDetail)
	for _, detail := range details {
		fieldUpdates[detail.EmployeePostID] = append(fieldUpdates[detail.EmployeePostID], detail)
	}

	// Update the maker table with approval info
	for empPostID, fields := range fieldUpdates {
		updateQuery := `UPDATE pmdm.post_mapping_detail_maker 
			SET approve_status = 'Approved', 
			    approved_date = $1, 
			    approved_by = $2, 
			    remarks = $3, 
			    updated_date = $4, 
			    updated_by = $5`
		updateArgs := []interface{}{time.Now(), fields[0].ApprovedBy, fields[0].Remarks, time.Now(), fields[0].ApprovedBy}

		argPos := 6
		for _, field := range fields {
			if field.FieldName != "" {
				updateQuery += `, ` + field.FieldName + ` = $` + strconv.Itoa(argPos)
				updateArgs = append(updateArgs, field.FieldValue)
				argPos++
			}
		}

		updateQuery += ` WHERE employee_post_id = $` + strconv.Itoa(argPos) + ` AND approve_status = 'Pending'`
		updateArgs = append(updateArgs, empPostID)

		cmdTag, err := tx.Exec(ctx, updateQuery, updateArgs...)
		if err != nil {
			return "", err
		}
		if cmdTag.RowsAffected() == 0 {
			return "", errors.New("no pending record found to approve for employee_post_id: " + strconv.Itoa(empPostID))
		}
	}

	// Insert or update into master table
	for empPostID, fields := range fieldUpdates {
		// Filter only approved fields
		insertColumns := []string{"employee_post_id", "employee_office_id", "updated_date", "updated_by"}
		insertValues := []interface{}{empPostID, fields[0].OfficeID, time.Now(), fields[0].ApprovedBy}
		setClauses := []string{}
		valueIndex := 5

		for _, field := range fields {
			if field.Status == "Rejected" {
				continue
			}
			insertColumns = append(insertColumns, field.FieldName)
			insertValues = append(insertValues, field.FieldValue)
			setClauses = append(setClauses, fmt.Sprintf("%s = EXCLUDED.%s", field.FieldName, field.FieldName))
			valueIndex++
		}

		if len(setClauses) == 0 {
			continue
		}

		insertQuery := fmt.Sprintf(`
			INSERT INTO pmdm.post_mapping_detail (%s)
			VALUES (%s)
			ON CONFLICT (employee_post_id) DO UPDATE SET
				employee_office_id = EXCLUDED.employee_office_id,
				updated_date = EXCLUDED.updated_date,
				updated_by = EXCLUDED.updated_by,
				%s
			RETURNING employee_post_id, employee_office_id
		`,
			strings.Join(insertColumns, ", "),
			strings.Join(generatePlaceholders(len(insertValues)), ", "),
			strings.Join(setClauses, ", "),
		)

		row := tx.QueryRow(ctx, insertQuery, insertValues...)

		var masterRecord domain.PosttoPostMap
		if err := row.Scan(&masterRecord.EmployeePostID, &masterRecord.OfficeID); err != nil {
			return "", err
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return "", err
	}

	return "record insert and update successfully", nil
}

// func (repo *PosttoPostMappingRepository) GetPostAndEmployeeHierarchy(ctx context.Context, officeID int) ([]domain.PostWithEmployee, error) {
// 	query := `
// 	WITH current_office AS (
// 		SELECT * FROM pmdm.kafka_office_hierarchy WHERE office_id = $1
// 	),
// 	relevant_offices AS (
// 		SELECT office_id FROM current_office
// 		UNION
// 		SELECT sub_division_office_id FROM current_office WHERE office_type_code = 'BPO'
// 		UNION
// 		SELECT division_office_id FROM current_office WHERE office_type_code IN ('BPO', 'SPO', 'HPO', 'DVN', 'PDN')
// 		UNION
// 		SELECT region_office_id FROM current_office WHERE office_type_code IN ('SPO', 'HPO', 'DVN', 'PDN', 'ADM')
// 		UNION
// 		SELECT circle_office_id FROM current_office WHERE office_type_code IN ('SPO', 'HPO', 'DVN', 'PDN', 'ADM')
// 	),
// 	head_posts AS (
// 		SELECT * FROM pmdm.post_management_master
// 		WHERE is_head_of_the_office = true
// 		AND office_id IN (SELECT office_id FROM relevant_offices)
// 	),
// 	group_posts AS (
// 		SELECT * FROM pmdm.post_management_master
// 		WHERE group_id IN (1, 2, 3)
// 		AND office_id IN (SELECT office_id FROM relevant_offices)
// 	),
// 	final_posts AS (
// 		SELECT * FROM head_posts
// 		UNION
// 		SELECT * FROM group_posts
// 	)
// 	SELECT
// 		fp.post_id,
// 		fp.post_name,
// 		fp.office_id,
// 		fp.office_name,
// 		fp.group_id,
// 		fp.group_name,
// 		fp.cadre_id,
// 		fp.cadre_name,
// 		fp.designation_id,
// 		fp.designation,
// 		kem.employee_id,
// 		COALESCE(
// 			kem.employee_first_name ||
// 			COALESCE(' ' || kem.employee_middle_name, '') ||
// 			COALESCE(' ' || kem.employee_last_name, ''),
// 			'Vacant'
// 		) AS employee_name
// 	FROM final_posts fp
// 	LEFT JOIN pmdm.kafka_employee_master kem
// 		ON kem.post_id = fp.post_id
// 		AND kem.office_id = fp.office_id
// 		AND kem.employment_status = 'Active'
// 	ORDER BY fp.office_name
// 	`

// 	rows, err := repo.db.Query(ctx, query, officeID)
// 	if err != nil {
// 		return nil, err
// 	}
// 	defer rows.Close()

// 	var results []domain.PostWithEmployee

// 	for rows.Next() {
// 		var p domain.PostWithEmployee
// 		err := rows.Scan(
// 			&p.PostID,
// 			&p.PostName,
// 			&p.OfficeID,
// 			&p.OfficeName,
// 			&p.GroupID,
// 			&p.GroupName,
// 			&p.CadreID,
// 			&p.CadreName,
// 			&p.DesignationID,
// 			&p.DesignationName,
// 			&p.EmployeeID,
// 			&p.EmployeeName,
// 		)
// 		if err != nil {
// 			return nil, err
// 		}
// 		results = append(results, p)
// 	}

//		return results, nil
//	}
// func (repo *PosttoPostMappingRepository) GetPostAndEmployeeHierarchy(ctx context.Context, officeID int) ([]domain.PostWithEmployee, error) {
// 	query := `
// WITH current_office AS (
//   SELECT * FROM pmdm.kafka_office_hierarchy WHERE office_id = $1
// ),
// related_offices AS (
//   SELECT office_id FROM current_office
//   UNION
//   SELECT circle_office_id FROM current_office WHERE circle_office_id IS NOT NULL
//   UNION
//   SELECT region_office_id FROM current_office WHERE region_office_id IS NOT NULL
//   UNION
//   SELECT division_office_id FROM current_office WHERE division_office_id IS NOT NULL
//   UNION
//   SELECT sub_division_office_id FROM current_office WHERE sub_division_office_id IS NOT NULL
//   UNION
//   SELECT ho_id FROM current_office WHERE ho_id IS NOT NULL
//   UNION
//   SELECT hro_id FROM current_office WHERE hro_id IS NOT NULL
//   UNION
//   SELECT 35320001  -- Always include this fixed office
// ),
// -- NEW: Add HO offices for head posts
// ho_offices AS (
//   SELECT DISTINCT ho_id as office_id
//   FROM pmdm.kafka_office_hierarchy
//   WHERE ho_id IS NOT NULL
//     AND (office_id = $1 OR
//          circle_office_id = $1 OR
//          region_office_id = $1 OR
//          division_office_id = $1 OR
//          sub_division_office_id = $1)
// ),
// -- NEW: Add Sub-Divisional offices for head posts
// subdiv_offices AS (
//   SELECT DISTINCT sub_division_office_id as office_id
//   FROM pmdm.kafka_office_hierarchy
//   WHERE sub_division_office_id IS NOT NULL
//     AND (office_id = $1 OR
//          circle_office_id = $1 OR
//          region_office_id = $1 OR
//          division_office_id = $1)
// ),
// -- Combine all office types
// all_related_offices AS (
//   SELECT office_id FROM related_offices
//   UNION
//   SELECT office_id FROM ho_offices WHERE office_id IS NOT NULL
//   UNION
//   SELECT office_id FROM subdiv_offices WHERE office_id IS NOT NULL
// ),
// head_posts AS (
//   SELECT * FROM pmdm.post_management_master
//   WHERE is_head_of_the_office = true
//     AND office_id IN (SELECT office_id FROM all_related_offices)
// ),
// group_posts AS (
//   SELECT * FROM pmdm.post_management_master
//   WHERE group_id IN (1, 2, 3)
//     AND office_id IN (SELECT office_id FROM all_related_offices)
// ),
// final_posts AS (
//   SELECT * FROM head_posts
//   UNION
//   SELECT * FROM group_posts
// )
// SELECT
//   fp.post_id,
//   fp.post_name,
//   fp.office_id,
//   fp.office_name,
//   fp.group_id,
//   fp.group_name,
//   fp.cadre_id,
//   fp.cadre_name,
//   fp.designation_id,
//   fp.designation,
//   kem.employee_id,
//   COALESCE(
//     kem.employee_first_name ||
//     COALESCE(' ' || kem.employee_middle_name, '') ||
//     COALESCE(' ' || kem.employee_last_name, ''),
//     'Vacant'
//   ) AS employee_name
// FROM final_posts fp
// LEFT JOIN pmdm.kafka_employee_master kem
//   ON kem.post_id = fp.post_id
//   AND kem.office_id = fp.office_id
//   AND kem.employment_status = 'Active'
// ORDER BY fp.office_name;

// WITH current_office AS (
// 	SELECT * FROM pmdm.kafka_office_hierarchy WHERE office_id = $1
//   ),
//   related_offices AS (
// 	SELECT office_id FROM current_office
// 	UNION
// 	SELECT circle_office_id FROM current_office WHERE circle_office_id IS NOT NULL
// 	UNION
// 	SELECT region_office_id FROM current_office WHERE region_office_id IS NOT NULL
// 	UNION
// 	SELECT division_office_id FROM current_office WHERE division_office_id IS NOT NULL
// 	UNION
// 	SELECT sub_division_office_id FROM current_office WHERE sub_division_office_id IS NOT NULL
// 	UNION
// 	SELECT ho_id FROM current_office WHERE ho_id IS NOT NULL
// 	UNION
// 	SELECT hro_id FROM current_office WHERE hro_id IS NOT NULL
// 	UNION
// 	SELECT 35320001  -- Always include this fixed office
//   ),
//   head_posts AS (
// 	SELECT * FROM pmdm.post_management_master
// 	WHERE is_head_of_the_office = true
// 	  AND office_id IN (SELECT office_id FROM related_offices)
//   ),
//   group_posts AS (
// 	SELECT * FROM pmdm.post_management_master
// 	WHERE group_id IN (1, 2, 3)
// 	  AND office_id IN (SELECT office_id FROM related_offices)
//   ),
//   final_posts AS (
// 	SELECT * FROM head_posts
// 	UNION
// 	SELECT * FROM group_posts
//   )
//   SELECT
// 	fp.post_id,
// 	fp.post_name,
// 	fp.office_id,
// 	fp.office_name,
// 	fp.group_id,
// 	fp.group_name,
// 	fp.cadre_id,
// 	fp.cadre_name,
// 	fp.designation_id,
// 	fp.designation,
// 	kem.employee_id,
// 	COALESCE(
// 	  kem.employee_first_name ||
// 	  COALESCE(' ' || kem.employee_middle_name, '') ||
// 	  COALESCE(' ' || kem.employee_last_name, ''),
// 	  'Vacant'
// 	) AS employee_name
//   FROM final_posts fp
//   LEFT JOIN pmdm.kafka_employee_master kem
// 	ON kem.post_id = fp.post_id
// 	AND kem.office_id = fp.office_id
// 	AND kem.employment_status = 'Active'
//   ORDER BY fp.office_name;
// 	rows, err := repo.db.Query(ctx, query, officeID)
// 	if err != nil {
// 		return nil, err
// 	}
// 	defer rows.Close()

// 	var results []domain.PostWithEmployee

// 	for rows.Next() {
// 		var p domain.PostWithEmployee
// 		err := rows.Scan(
// 			&p.PostID,
// 			&p.PostName,
// 			&p.OfficeID,
// 			&p.OfficeName,
// 			&p.GroupID,
// 			&p.GroupName,
// 			&p.CadreID,
// 			&p.CadreName,
// 			&p.DesignationID,
// 			&p.DesignationName,
// 			&p.EmployeeID,
// 			&p.EmployeeName,
// 		)
// 		if err != nil {
// 			return nil, err
// 		}
// 		results = append(results, p)
// 	}

// 	return results, nil
// }

func (s *PosttoPostMappingRepository) SavePostMappings(ctx context.Context, payload domain.PostMappingPayload) error {
	tx, err := s.db.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	// Whitelisted columns (exact table fields to avoid SQL injection)
	allowedColumns := map[string]bool{
		"gds_leave_sanc_authority_1":      true,
		"gds_leave_sanc_authority_2":      true,
		"reporting_authority":             true,
		"apar_reporting_authority":        true,
		"apar_review_authority":           true,
		"apar_controlling_authority":      true,
		"apar_appellate_authority":        true,
		"service_book_approve_authority1": true,
		"leave_sanc_authority_1":          true,
		"leave_sanc_authority_2":          true,
		"leave_sanc_authority_3":          true,
		"pay_approve_authority1":          true,
		"pay_approve_authority2":          true,
		"leave_fwd_authority1":            true,
		"leave_fwd_authority2":            true,
		"pay_fwd_authority1":              true,
		"pay_fwd_authority2":              true,
		"appointing_authority":            true,
		"disciplinary_authority":          true,
		"ddo_authority":                   true,
		"admin_authority":                 true,
		"pension_sanctioning_authority":   true,
		"pension_authorising_authority":   true,
		"service_book_approve_authority2": true,
		"role_authority":                  true,
		"service_book_foward_authority1":  true,
		"service_book_foward_authority2":  true,
		"vigilence_maker_authority":       true,
		"admin_office":                    true,
		"apar_custodian_authority":        true,
	}

	for _, entry := range payload.MappingData {
		var exists bool
		err := tx.QueryRow(ctx, `SELECT EXISTS (SELECT 1 FROM pmdm.post_mapping_detail WHERE employee_post_id = $1)`, entry.PostID).Scan(&exists)
		if err != nil {
			return err
		}

		var setClauses []string
		var args []interface{}
		argPos := 1

		for col, val := range entry.Changes {
			if !allowedColumns[col] {
				continue // skip unknown/unsafe columns
			}
			setClauses = append(setClauses, fmt.Sprintf("%s = $%d", col, argPos))
			args = append(args, val)
			argPos++
		}

		setClauses = append(setClauses, fmt.Sprintf("updated_by = $%d", argPos))
		args = append(args, payload.Meta.CreatedBy)
		argPos++

		setClauses = append(setClauses, fmt.Sprintf("updated_date = $%d", argPos))
		args = append(args, time.Now())
		argPos++

		if exists {
			query := fmt.Sprintf(`UPDATE pmdm.post_mapping_detail SET %s WHERE employee_post_id = $%d`,
				strings.Join(setClauses, ", "), argPos)
			args = append(args, entry.PostID)
			fmt.Println(query)
			_, err = tx.Exec(ctx, query, args...)
		} else {
			columns := []string{"employee_post_id", "employee_office_id", "updated_by", "updated_date"}
			insertArgs := []interface{}{entry.PostID, entry.OfficeID, payload.Meta.CreatedBy, time.Now()}

			for col, val := range entry.Changes {
				if !allowedColumns[col] {
					continue
				}
				columns = append(columns, col)
				insertArgs = append(insertArgs, val)
			}

			placeholders := make([]string, len(columns))
			for i := range placeholders {
				placeholders[i] = fmt.Sprintf("$%d", i+1)
			}

			query := fmt.Sprintf(`INSERT INTO pmdm.post_mapping_detail (%s) VALUES (%s)`,
				strings.Join(columns, ", "), strings.Join(placeholders, ", "))
			_, err = tx.Exec(ctx, query, insertArgs...)
		}

		if err != nil {
			return err
		}
	}

	return tx.Commit(ctx)
}
func (s *PosttoPostMappingRepository) SavePostMappingsRepo(ctx context.Context, payload domain.PostMappingPayload) error {

	const batchSize = 50

	allowedColumns := map[string]bool{
		"gds_leave_sanc_authority_1":      true,
		"gds_leave_sanc_authority_2":      true,
		"reporting_authority":             true,
		"apar_reporting_authority":        true,
		"apar_review_authority":           true,
		"apar_controlling_authority":      true,
		"apar_appellate_authority":        true,
		"service_book_approve_authority1": true,
		"leave_sanc_authority_1":          true,
		"leave_sanc_authority_2":          true,
		"leave_sanc_authority_3":          true,
		"pay_approve_authority1":          true,
		"pay_approve_authority2":          true,
		"leave_fwd_authority1":            true,
		"leave_fwd_authority2":            true,
		"pay_fwd_authority1":              true,
		"pay_fwd_authority2":              true,
		"appointing_authority":            true,
		"disciplinary_authority":          true,
		"ddo_authority":                   true,
		"admin_authority":                 true,
		"pension_sanctioning_authority":   true,
		"pension_authorising_authority":   true,
		"service_book_approve_authority2": true,
		"role_authority":                  true,
		"service_book_foward_authority1":  true,
		"service_book_foward_authority2":  true,
		"vigilence_maker_authority":       true,
		"admin_office":                    true,
		"apar_custodian_authority":        true,
	}

	tx, err := s.db.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	total := len(payload.MappingData)

	for start := 0; start < total; start += batchSize {
		end := start + batchSize
		if end > total {
			end = total
		}
		batch := payload.MappingData[start:end]

		// Base columns
		baseColumns := []string{"employee_post_id", "employee_office_id", "updated_by", "updated_date"}
		extraColumnsSet := make(map[string]struct{})

		// Collect distinct allowed columns present in batch
		for _, entry := range batch {
			for col := range entry.Changes {
				if allowedColumns[col] {
					extraColumnsSet[col] = struct{}{}
				}
			}
		}

		// Deterministic order
		var extraColumns []string
		for col := range extraColumnsSet {
			extraColumns = append(extraColumns, col)
		}
		sort.Strings(extraColumns)

		// Full set of columns
		allColumns := append(baseColumns, extraColumns...)

		// Values
		valueStrings := make([]string, 0, len(batch))
		valueArgs := make([]interface{}, 0, len(allColumns)*len(batch))
		argPos := 1

		for _, entry := range batch {
			rowValues := make([]string, 0, len(allColumns))

			// base columns
			rowValues = append(rowValues, fmt.Sprintf("$%d", argPos))
			valueArgs = append(valueArgs, entry.PostID)
			argPos++

			rowValues = append(rowValues, fmt.Sprintf("$%d", argPos))
			valueArgs = append(valueArgs, entry.OfficeID)
			argPos++

			rowValues = append(rowValues, fmt.Sprintf("$%d", argPos))
			valueArgs = append(valueArgs, payload.Meta.CreatedBy)
			argPos++

			rowValues = append(rowValues, fmt.Sprintf("$%d", argPos))
			valueArgs = append(valueArgs, time.Now())
			argPos++

			// extra allowed columns
			for _, col := range extraColumns {
				if val, ok := entry.Changes[col]; ok {
					rowValues = append(rowValues, fmt.Sprintf("$%d", argPos))
					valueArgs = append(valueArgs, val)
				} else {
					rowValues = append(rowValues, fmt.Sprintf("$%d", argPos))
					valueArgs = append(valueArgs, nil) // excluded col will be NULL
				}
				argPos++
			}

			valueStrings = append(valueStrings, fmt.Sprintf("(%s)", strings.Join(rowValues, ", ")))
		}

		// Updates → COALESCE to keep old value if EXCLUDED.col is NULL
		updates := make([]string, 0, len(extraColumns)+2)
		for _, col := range extraColumns {
			updates = append(updates,
				fmt.Sprintf("%s = COALESCE(EXCLUDED.%s, pmdm.post_mapping_detail.%s)", col, col, col))
		}

		// audit columns always updated
		updates = append(updates,
			"updated_by = EXCLUDED.updated_by",
			"updated_date = EXCLUDED.updated_date",
		)

		query := fmt.Sprintf(`
			INSERT INTO pmdm.post_mapping_detail (%s) VALUES %s
			ON CONFLICT (employee_post_id) DO UPDATE SET %s`,
			strings.Join(allColumns, ", "),
			strings.Join(valueStrings, ", "),
			strings.Join(updates, ", "),
		)

		if _, err := tx.Exec(ctx, query, valueArgs...); err != nil {
			return err
		}
	}

	return tx.Commit(ctx)
}

func (rmr *PosttoPostMappingRepository) IdentifyHeadOfOffice(gctx *gin.Context, officeID int64) ([]domain.HeadOfOfficeEntry, error) {
	ctx, cancel := context.WithTimeout(gctx, 10*time.Second)
	defer cancel()

	query := `
	WITH cadre_ranked AS (
		SELECT
			pm.postmanagement_id,
			pm.office_id,
			pm.post_id,
			pm.post_name,
			pm.cadre_name,
			ROW_NUMBER() OVER (
				PARTITION BY pm.office_id
				ORDER BY co.cadre_order
			) AS rn_cadre
		FROM pmdm.post_management_master pm
		LEFT JOIN pmdm.cadre_order co
			ON TRIM(pm.cadre_name) = TRIM(co.cadre_name)
		WHERE pm.office_id = $1
	),
	pay_base AS (
		SELECT 
			a.employee_id, 
			a.level, 
			a.index, 
			b.post_id
		FROM pmdm.kafka_pay_basicpay a
		JOIN (
			SELECT employee_id, post_id 
			FROM pmdm.kafka_employee_master 
			WHERE employment_status = 'Active'
		) b ON a.employee_id = b.employee_id
		WHERE CURRENT_DATE BETWEEN a.valid_from_date AND a.valid_to_date
	),
	pay_ranked AS (
		SELECT
			pm.postmanagement_id,
			pm.office_id,
			ROW_NUMBER() OVER (
				PARTITION BY pm.office_id
				ORDER BY pb.level DESC, pb.index DESC
			) AS rn_pay
		FROM pmdm.post_management_master pm
		JOIN pay_base pb
			ON pm.post_id = pb.post_id
		WHERE pm.office_id = $1
	),
	final_ranked AS (
		SELECT
			cr.postmanagement_id,
			cr.office_id,
			cr.post_id,
			cr.post_name,
			cr.cadre_name,
			cr.rn_cadre,
			pr.rn_pay
		FROM cadre_ranked cr
		LEFT JOIN pay_ranked pr
			ON cr.postmanagement_id = pr.postmanagement_id
	)
	SELECT 
		postmanagement_id,
		office_id,
		post_id,
		post_name,
		cadre_name,
		rn_cadre,
		rn_pay
	FROM final_ranked
	WHERE rn_cadre = 1
	   OR (rn_cadre IS NULL AND rn_pay = 1)
	`

	rows, err := rmr.db.Query(ctx, query, officeID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var result []domain.HeadOfOfficeEntry
	for rows.Next() {
		var (
			r       domain.HeadOfOfficeEntry
			rnCadre sql.NullInt64
			rnPay   sql.NullInt64
		)

		err := rows.Scan(
			&r.PostManagementID,
			&r.OfficeID,
			&r.PostID,
			&r.PostName,
			&r.CadreName,
			&rnCadre,
			&rnPay,
		)
		if err != nil {
			return nil, err
		}

		if rnCadre.Valid {
			v := int(rnCadre.Int64)
			r.RankByCadre = &v
		}
		if rnPay.Valid {
			v := int(rnPay.Int64)
			r.RankByPay = &v
		}

		result = append(result, r)
	}

	return result, nil
}
func (rmr *PosttoPostMappingRepository) UpdateHeadOfOffice(
	gctx *gin.Context,
	officeID, newPostID, callerPostID int64) error {

	ctx, cancel := context.WithTimeout(gctx, 10*time.Second)
	defer cancel()

	const authSQL = `
	WITH allowed AS (
		SELECT office_id, post_id
		FROM pmdm.post_management_master
		WHERE office_id IN (         -- offices reporting to caller’s office
			SELECT office_id
			FROM pmdm.kafka_office_master
			WHERE reporting_office_id = (
				SELECT office_id
				FROM pmdm.post_management_master
				WHERE post_id = $1
				  AND is_head_of_the_office = TRUE
			)
		)
		UNION ALL                    -- caller’s own office
		SELECT office_id, post_id
		FROM pmdm.post_management_master
		WHERE office_id = (
			SELECT office_id
			FROM pmdm.post_management_master
			WHERE post_id = $1
			  AND is_head_of_the_office = TRUE
		)
	)
	SELECT 1
	FROM   allowed
	WHERE  office_id = $2
	  AND  post_id   = $3;
	`

	var ok int
	if err := rmr.db.
		QueryRow(ctx, authSQL, callerPostID, officeID, newPostID).
		Scan(&ok); err != nil {
		// No rows → unauthorised
		return fmt.Errorf("unauthorised: post %d cannot update office %d", callerPostID, officeID)
	}
	tx, err := rmr.db.Begin(ctx)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback(ctx) }() // protects against panic / error

	// a) clear existing HoO
	if _, err = tx.Exec(ctx, `
		UPDATE pmdm.post_management_master
		SET    is_head_of_the_office = FALSE
		WHERE  office_id = $1`,
		officeID); err != nil {
		return err
	}

	// b) set new HoO
	cmdTag, err := tx.Exec(ctx, `
		UPDATE pmdm.post_management_master
		SET    is_head_of_the_office = TRUE
		WHERE  post_id = $1 AND office_id = $2`,
		newPostID, officeID)
	if err != nil {
		return err
	}
	if cmdTag.RowsAffected() == 0 {
		return fmt.Errorf("post %d does not belong to office %d", newPostID, officeID)
	}

	return tx.Commit(ctx)
}
func (rmr *PosttoPostMappingRepository) GetHeadPostOccupant(gctx *gin.Context, officeID int64) (domain.HeadPostOccupancy, error) {
	ctx, cancel := context.WithTimeout(gctx, 5*time.Second)
	defer cancel()

	query := `
	SELECT
		COALESCE(em.employee_id, 0) AS employee_id,
		COALESCE(
			NULLIF(CONCAT_WS(' ',
				em.employee_first_name,
				em.employee_middle_name,
				em.employee_last_name), ''), 'Vacant'
		) AS employee_name,
		pm.post_id,
		pm.office_id,
		pm.post_name,
		pm.office_name,
		pm.group_name,
		pm.cadre_name,
		pm.status
	FROM pmdm.post_management_master pm
	LEFT JOIN pmdm.kafka_employee_master em
		ON pm.post_id = em.post_id AND em.employment_status = 'Active'
	WHERE pm.office_id = $1
	  AND pm.status = 'Active'
	  AND pm.is_head_of_the_office = TRUE
	LIMIT 1;`

	var (
		employeeID   int64
		employeeName sql.NullString
		postID       int64
		officeIDOut  int64
		postName     sql.NullString
		officeName   sql.NullString
		groupName    sql.NullString
		cadreName    sql.NullString
		status       sql.NullString
	)

	err := rmr.db.QueryRow(ctx, query, officeID).Scan(
		&employeeID,
		&employeeName,
		&postID,
		&officeIDOut,
		&postName,
		&officeName,
		&groupName,
		&cadreName,
		&status,
	)
	if err != nil {
		return domain.HeadPostOccupancy{}, fmt.Errorf("error fetching head post occupant: %w", err)
	}

	result := domain.HeadPostOccupancy{
		EmployeeID:   employeeID,
		EmployeeName: getSafeString(employeeName, "Vacant"),
		PostID:       postID,
		OfficeID:     officeIDOut,
		PostName:     getSafeString(postName, ""),
		OfficeName:   getSafeString(officeName, ""),
		GroupName:    getSafeString(groupName, ""),
		CadreName:    getSafeString(cadreName, ""),
		Status:       getSafeString(status, ""),
	}

	return result, nil
}

func getSafeString(ns sql.NullString, fallback string) string {
	if ns.Valid {
		return ns.String
	}
	return fallback
}

func (rmr *PosttoPostMappingRepository) GetPostDetailsForHOO(gctx *gin.Context, officeID int64) ([]domain.HeadPostOccupancy, error) {
	ctx, cancel := context.WithTimeout(gctx, 5*time.Second)
	defer cancel()

	query := `
	SELECT
		COALESCE(em.employee_id, 0) AS employee_id,
		COALESCE(
			NULLIF(CONCAT_WS(' ',
				em.employee_first_name,
				em.employee_middle_name,
				em.employee_last_name), ''), 'Vacant'
		) AS employee_name,
		pm.post_id,
		pm.office_id,
		pm.post_name,
		pm.office_name,
		pm.group_name,
		pm.cadre_name,
		pm.status
	FROM pmdm.post_management_master pm
	LEFT JOIN pmdm.kafka_employee_master em
		ON pm.post_id = em.post_id
	WHERE pm.office_id = $1
	  AND pm.status = 'Active';
	`

	rows, err := rmr.db.Query(ctx, query, officeID)
	if err != nil {
		return nil, fmt.Errorf("error executing query: %w", err)
	}
	defer rows.Close()

	var results []domain.HeadPostOccupancy

	for rows.Next() {
		var (
			employeeID   int64
			employeeName sql.NullString
			postID       int64
			officeIDOut  int64
			postName     sql.NullString
			officeName   sql.NullString
			groupName    sql.NullString
			cadreName    sql.NullString
			status       sql.NullString
		)

		if err := rows.Scan(
			&employeeID,
			&employeeName,
			&postID,
			&officeIDOut,
			&postName,
			&officeName,
			&groupName,
			&cadreName,
			&status,
		); err != nil {
			return nil, fmt.Errorf("error scanning row: %w", err)
		}

		result := domain.HeadPostOccupancy{
			EmployeeID:   employeeID,
			EmployeeName: getSafeString(employeeName, "Vacant"),
			PostID:       postID,
			OfficeID:     officeIDOut,
			PostName:     getSafeString(postName, ""),
			OfficeName:   getSafeString(officeName, ""),
			GroupName:    getSafeString(groupName, ""),
			CadreName:    getSafeString(cadreName, ""),
			Status:       getSafeString(status, ""),
		}

		results = append(results, result)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error during row iteration: %w", err)
	}

	return results, nil
}

func (rmr *PosttoPostMappingRepository) GetPostRedeplomentByOfficeID(ctx *gin.Context, officeID int, cadreName string) ([]domain.PostRedeployment, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.
		Select(`h.office_id as office_id,
    	h.office_name as office_name,
		h.office_type_code as office_type,
    	pmm.post_id as post_id,
    	pmm.post_name as post_name,
    	pmm.cadre_name as cadre_name,
    	pmm.designation as designation,
    	COALESCE(
        NULLIF(
            TRIM(CONCAT_WS(' ',
                em.employee_first_name,
                em.employee_middle_name,
                em.employee_last_name
            	)), ''
        	),
       	 'Vacant'
    	) AS "employee_name",
    	CASE
        WHEN em.employee_id IS NOT NULL THEN 'Filled'
        ELSE 'Vacant'
    	END AS filled_status`).
		From("pmdm.kafka_office_hierarchy h").
		Join("pmdm.post_management_master pmm ON pmm.office_id = h.office_id").
		LeftJoin(`pmdm.kafka_employee_master em 
                 ON em.post_id = pmm.post_id 
                 AND em.employment_status = 'Active'`).
		Where("h.division_office_id = ?", officeID). // Use parameterized query
		Where("pmm.cadre_name = ?", cadreName).
		OrderBy(`office_name, post_name`)

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.PostRedeployment])
}

func (rmr *PosttoPostMappingRepository) CheckOfficeCompatibility(ctx *gin.Context, postID int, officeID int) (bool, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	// Build the combined query
	query := dblib.Psql.Select("COUNT(*)").
		From("pmdm.post_management_master pmm").
		Join("pmdm.officetype_cadre_mapping ocm ON pmm.cadre_name = ocm.cadre_name").
		Join("pmdm.kafka_office_master kom ON ocm.office_type_id = kom.office_type_id").
		Where(sq.And{
			sq.Eq{"pmm.post_id": postID},
			sq.Eq{"kom.office_id": officeID},
		})

	// Convert to executable SQL
	sql, args, err := query.ToSql()
	if err != nil {
		return false, fmt.Errorf("failed to build compatibility query: %w", err)
	}

	// Execute and get result
	var count int
	err = rmr.db.QueryRow(gctx, sql, args...).Scan(&count)
	if err != nil {
		return false, fmt.Errorf("failed to check compatibility: %w", err)
	}
	return count > 0, nil
}

// func (rmr *PosttoPostMappingRepository) SavePostRedeploymentRepo(ctx *gin.Context, value domain.PostRedeploymentLog,
// 	value1 domain.PostManagementMasterUpdate) ([]domain.PostRedeploymentLog, error) {
// 	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
// 	defer cancel()
// 	batch := &pgx.Batch{}

// 	// Insert into post_redeployment_log
// 	query := dblib.Psql.Insert("pmdm.post_redeployment_log").
// 		Columns(
// 			"post_id",
// 			"cadre_name",
// 			"redeployment_from_office_id",
// 			"redeployment_to_office_id",
// 			"redeployment_on",
// 			"redeployment_by",
// 			"effective_from",
// 			"effective_upto",
// 		).
// 		Values(
// 			value.PostID,
// 			value.CadreName,
// 			value.RedeploymentFromOfficeID,
// 			value.RedeploymentToOfficeID,
// 			value.RedeploymentOn,
// 			value.RedeploymentBy,
// 			value.EffectiveFrom,
// 			value.EffectiveUPTo,
// 		)

// 	if err := dblib.QueueExecRow(batch, query); err != nil {
// 		return nil, err
// 	}

// 	// Update post_management_master using value1 for all master fields
// 	uquery := dblib.Psql.Update("pmdm.post_management_master").
// 		Set("Office_id", value1.OfficeID).
// 		Set("Office_Name", value1.OfficeName).
// 		Set("Filled_Status", value1.FilledStatus).
// 		Set("Post_Status", value1.PostStatus).
// 		Set("Allowances_Attached", value1.AllowancesAttached).
// 		Set("Allowance_Description", value1.AllowanceDescription).
// 		Set("Updated_By", value1.UpdatedBy).
// 		Set("Updated_Date", value1.UpdatedDate).
// 		Set("Status", value1.Status).
// 		Set("Remarks", value1.Remarks).
// 		Set("Valid_From", value1.ValidFrom).
// 		Set("Valid_To", value1.ValidTo).
// 		Set("Order_Casemark", value1.OrderCasemark).
// 		Set("Order_Date", value1.OrderDate).
// 		Set("Upload_Order_Doc_Name", value1.UploadOrderDocName).
// 		Set("Establishment_Register_ID", value1.EstablishmentRegisterID).
// 		Set("Designation", value1.Designation).
// 		Set("Permanent_Status", value1.PermanentStatus).
// 		Set("Establishment_Register_Name", value1.EstablishmentRegisterName).
// 		Set("Group_Name", value1.GroupName).
// 		Set("Office_Type", value1.OfficeType).
// 		Set("Office_Supervisor", value1.OfficeSupervisor).
// 		Set("Is_Head_of_the_Office", value1.IsHeadOfTheOffice).
// 		Set("Cadre_ID", value1.CadreID).
// 		Set("Designation_ID", value1.DesignationID).
// 		Set("Post_Name", value1.PostName).
// 		Where(sq.And{
// 			sq.Eq{"post_id": value.PostID},
// 			sq.Eq{"office_id": value.RedeploymentFromOfficeID},
// 		})

// 	if err := dblib.QueueExecRow(batch, uquery); err != nil {
// 		return nil, err
// 	}

// 	if err := rmr.db.SendBatch(gctx, batch).Close(); err != nil {
// 		log.Error(gctx, "Error while redeploying the posts: %v", err)
// 		return nil, err
// 	}

// 	return []domain.PostRedeploymentLog{value}, nil
// }

func (rmr *PosttoPostMappingRepository) GetCircleOfficeIDs(ctx *gin.Context) ([]domain.CircleName, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.Select("DISTINCT circle_office_id, circle_name").
		From("pmdm.kafka_office_hierarchy")

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.CircleName])
}

func (rmr *PosttoPostMappingRepository) GetRegionalOfficeIDs(ctx *gin.Context, circleOfficeID int) ([]domain.RegionName, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.Select(`DISTINCT 
        CASE 
            WHEN region_office_id = '0' THEN circle_office_id 
            ELSE region_office_id 
        END AS region_office_id`,
		`CASE 
            WHEN region_office_id IN ('0') THEN circle_name 
            ELSE region_name 
        END AS region_name`).
		From("pmdm.kafka_office_hierarchy").
		Where("circle_office_id = ?", circleOfficeID)

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.RegionName])
}

func (rmr *PosttoPostMappingRepository) GetDivisionalOfficeIDs(ctx *gin.Context, regionalofficeid int) ([]domain.DivisionName, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.Select(`DISTINCT 
        CASE 
            WHEN division_office_id = '0' THEN circle_office_id 
            ELSE division_office_id 
        END AS division_office_id`,
		`CASE 
            WHEN division_office_id IN ('0') THEN circle_name 
            ELSE division_name 
        END AS division_name`).
		From("pmdm.kafka_office_hierarchy").
		Where("region_office_id = ?", regionalofficeid)

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.DivisionName])
}

func (rmr *PosttoPostMappingRepository) GetCadreDetails(ctx *gin.Context) ([]domain.CadreName, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.Select("DISTINCT cadre_id, cadre_name").
		From("pmdm.cadre_master")

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.CadreName])
}

func (rmr *PosttoPostMappingRepository) GetPostAndEmployeeHierarchy(ctx *gin.Context, officeID int) ([]domain.PostWithEmployee, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := `WITH current_office AS (SELECT * FROM pmdm.kafka_office_hierarchy WHERE office_id = $1), 
	related_offices AS (SELECT unnest(array[
	office_id, circle_office_id, region_office_id, division_office_id, sub_division_office_id, ho_id, hro_id,sro_id,35320001]) AS office_id
  	FROM current_office),
	ho_offices AS (SELECT DISTINCT ho_id AS office_id 
  	FROM pmdm.kafka_office_hierarchy 
  	WHERE ho_id IS NOT NULL
    AND $1 IN (office_id, circle_office_id,  region_office_id, division_office_id, sub_division_office_id)),
	subdiv_offices AS (
  	SELECT DISTINCT sub_division_office_id AS office_id 
  	FROM pmdm.kafka_office_hierarchy 
 	WHERE sub_division_office_id IS NOT NULL
    AND $1 IN (office_id, circle_office_id, region_office_id, division_office_id)),
	all_related_offices AS (
  	SELECT DISTINCT office_id FROM related_offices
  	UNION
  	SELECT office_id FROM ho_offices
  	UNION
  	SELECT office_id FROM subdiv_offices),
	head_posts AS (
  	SELECT * 
  	FROM pmdm.post_management_master
  	WHERE is_head_of_the_office = true
    AND office_id IN (SELECT office_id FROM all_related_offices)
	), 
	group_posts AS (
  	SELECT * 
  	FROM pmdm.post_management_master
  	WHERE group_id IN (1, 2, 3, 4)
    AND office_id IN (SELECT office_id FROM all_related_offices)
	), 
	final_posts AS (
  	SELECT * FROM head_posts
  	UNION
  	SELECT * FROM group_posts
	) 
	SELECT
  	fp.post_id,
  	fp.post_name,
  	fp.office_id,
  	fp.office_name,
  	fp.group_id,
  	fp.group_name,
  	fp.cadre_id,
  	fp.cadre_name,
  	fp.designation_id,
  	fp.designation,
  	kem.employee_id,
  	COALESCE(
    kem.employee_first_name || 
    COALESCE(' ' || kem.employee_middle_name, '') || 
    COALESCE(' ' || kem.employee_last_name, ''),
    'Vacant'
  	) AS employee_name
	FROM final_posts fp
	LEFT JOIN pmdm.kafka_employee_master kem
  	ON kem.post_id = fp.post_id
  	AND kem.office_id = fp.office_id
  	AND kem.employment_status = 'Active'
	ORDER BY fp.office_name;`
	rows, err := rmr.db.Query(gctx, query, officeID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []domain.PostWithEmployee

	for rows.Next() {
		var p domain.PostWithEmployee
		err := rows.Scan(
			&p.PostID,
			&p.PostName,
			&p.OfficeID,
			&p.OfficeName,
			&p.GroupID,
			&p.GroupName,
			&p.CadreID,
			&p.CadreName,
			&p.DesignationID,
			&p.DesignationName,
			&p.EmployeeID,
			&p.EmployeeName,
		)
		if err != nil {
			return nil, err
		}
		results = append(results, p)
	}

	return results, nil
}

func (rmr *PosttoPostMappingRepository) GetPostDetailsForRedeployment(ctx *gin.Context, postid int) ([]domain.PostDetailsForRedeployment, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.Select(
		"h.office_id AS office_id",
		"h.office_name AS office_name",
		"h.office_type_code AS office_type",
		"pmm.post_id AS post_id",
		"pmm.post_name AS post_name",
		"pmm.cadre_name AS cadre_name",
		"pmm.designation AS designation",
		"COALESCE(NULLIF(TRIM(CONCAT_WS(' ', em.employee_first_name, em.employee_middle_name, em.employee_last_name)), ''), 'Vacant') AS employee_name",
		"CASE WHEN em.employee_id IS NOT NULL THEN 'Filled' ELSE 'Vacant' END AS filled_status",
		"pmm.Post_Status",
		"pmm.Allowances_Attached",
		"pmm.Allowance_Description",
		"pmm.Updated_By",
		"pmm.Updated_Date",
		"pmm.Status",
		"pmm.Remarks",
		"pmm.Valid_From",
		"pmm.Valid_To",
		"pmm.Order_Casemark",
		"pmm.Order_Date",
		"pmm.Upload_Order_Doc_Name",
		"pmm.Establishment_Register_ID",
		"pmm.Permanent_Status",
		"pmm.Establishment_Register_Name",
		"pmm.Group_Name",
		"pmm.Office_Supervisor",
		"pmm.Is_Head_of_the_Office",
		"pmm.group_id",
		"pmm.cadre_id",
		"pmm.designation_id",
	).
		From("pmdm.kafka_office_hierarchy h").
		Join("pmdm.post_management_master pmm ON pmm.office_id = h.office_id").
		LeftJoin("pmdm.kafka_employee_master em ON em.post_id = pmm.post_id AND em.employment_status = 'Active'").
		Where(sq.Eq{"pmm.post_id": postid}).
		OrderBy("office_name", "post_name")

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.PostDetailsForRedeployment])
}

func (rmr *PosttoPostMappingRepository) GetPostDetails(ctx *gin.Context) ([]domain.PostNameDetails, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.
		Select("post_id, MIN(post_name) AS post_name").
		From("pmdm.post_management_master").
		Where(sq.And{
			sq.NotEq{"post_name": nil},       // Exclude NULLs
			sq.Expr("TRIM(post_name) <> ''"), // Exclude empty/blank names
		}).
		GroupBy("post_id")

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.PostNameDetails])
}

func (rmr *PosttoPostMappingRepository) GetDesignationDetails2(ctx *gin.Context) ([]domain.DesignationDetails, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.
		Select("DISTINCT designation_id, designation").
		From("pmdm.designation_master").
		OrderBy("designation")

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.DesignationDetails])
}

func (rmr *PosttoPostMappingRepository) GetDesignationDetails(ctx *gin.Context) ([]domain.DesignationDetails, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.
		Select("designation_id, MIN(designation) AS designation").
		From("pmdm.post_management_master").
		Where(sq.And{
			sq.NotEq{"designation": nil},
			sq.Expr("TRIM(designation) <> ''"),
			sq.NotEq{"designation_id": nil},
		}).
		GroupBy("designation_id")

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.DesignationDetails])
}

func (rmr *PosttoPostMappingRepository) GetPostRedeployedInwardReports(ctx *gin.Context, officeID int) ([]domain.PostRedeploymentReport, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.
		Select(
			"DISTINCT post_id",
			"cadre_name",
			"redeployment_from_office_id",
			"redeployment_to_office_id",
			"redeployment_on",
			"redeployment_by",
			"effective_from",
			"effective_upto",
			"upload_order_doc_name",
		).
		From("pmdm.post_redeployment_log").
		Where("redeployment_to_office_id IN (SELECT office_id FROM kafka_office_hierarchy WHERE circle_office_id = ?)",
			officeID)

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.PostRedeploymentReport])
}

func (rmr *PosttoPostMappingRepository) GetPostRedeployedOutwardReports(ctx *gin.Context, officeID int) ([]domain.PostRedeploymentReport, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.
		Select(
			"DISTINCT post_id",
			"cadre_name",
			"redeployment_from_office_id",
			"redeployment_to_office_id",
			"redeployment_on",
			"redeployment_by",
			"effective_from",
			"effective_upto",
			"upload_order_doc_name",
		).
		From("pmdm.post_redeployment_log").
		Where("redeployment_from_office_id IN (SELECT office_id FROM kafka_office_hierarchy WHERE circle_office_id = ?)",
			officeID)

	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.PostRedeploymentReport])
}

func (rmr *PosttoPostMappingRepository) SavePostRedeploymentRepo2(ctx *gin.Context, tx pgx.Tx, value domain.PostRedeploymentLog,
	value1 domain.PostManagementMasterUpdate, filename string) error {

	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	batch := &pgx.Batch{}

	// Insert into post_redeployment_log
	query := dblib.Psql.Insert("pmdm.post_redeployment_log").
		Columns(
			"post_id",
			"cadre_name",
			"redeployment_from_office_id",
			"redeployment_to_office_id",
			"redeployment_on",
			"redeployment_by",
			"effective_from",
			"effective_upto",
			"upload_order_doc_name",
		).
		Values(
			value.PostID,
			value.CadreName,
			value.RedeploymentFromOfficeID,
			value.RedeploymentToOfficeID,
			value.RedeploymentOn,
			value.RedeploymentBy,
			value.EffectiveFrom,
			value.EffectiveUPTo,
			filename,
		)

	if err := dblib.QueueExecRow(batch, query); err != nil {
		return err
	}
	// Check and update existing head of office if value1.IsHeadOfTheOffice is true
	if value1.IsHeadOfTheOffice {
		// Check if another post has Is_Head_of_the_Office = true for same office
		row := tx.QueryRow(gctx, `
			SELECT post_id FROM pmdm.post_management_master 
			WHERE office_id = $1 AND is_head_of_the_office = true`, value.RedeploymentToOfficeID)

		var prevHeadPostID int
		err := row.Scan(&prevHeadPostID)
		if err == nil {
			// Previous head exists, update is_head_of_the_office to false
			headUpdate := dblib.Psql.Update("pmdm.post_management_master").
				Set("Is_Head_of_the_Office", false).
				Where(sq.Eq{"post_id": prevHeadPostID})

			if err := dblib.QueueExecRow(batch, headUpdate); err != nil {
				return fmt.Errorf("failed to queue head office flag removal: %w", err)
			}
		} else if err != pgx.ErrNoRows {
			return fmt.Errorf("error checking existing head of office: %w", err)
		}
	}
	// Update post_management_master using value1 for all master fields
	uquery := dblib.Psql.Update("pmdm.post_management_master").
		Set("Office_id", value1.OfficeID).
		Set("Office_Name", value1.OfficeName).
		Set("Filled_Status", value1.FilledStatus).
		Set("Post_Status", value1.PostStatus).
		Set("Allowances_Attached", value1.AllowancesAttached).
		Set("Allowance_Description", value1.AllowanceDescription).
		Set("Updated_By", value1.UpdatedBy).
		Set("Updated_Date", value1.UpdatedDate).
		Set("Status", value1.Status).
		Set("Remarks", value1.Remarks).
		Set("Valid_From", value1.ValidFrom).
		Set("Valid_To", value1.ValidTo).
		Set("Order_Casemark", value1.OrderCasemark).
		Set("Order_Date", value1.OrderDate).
		Set("Upload_Order_Doc_Name", filename).
		Set("Establishment_Register_ID", value1.EstablishmentRegisterID).
		Set("Designation", value1.Designation).
		Set("Permanent_Status", value1.PermanentStatus).
		Set("Establishment_Register_Name", value1.EstablishmentRegisterName).
		Set("Group_Name", value1.GroupName).
		Set("Office_Type", value1.OfficeType).
		Set("Office_Supervisor", value1.OfficeSupervisor).
		Set("Is_Head_of_the_Office", value1.IsHeadOfTheOffice).
		Set("Cadre_ID", value1.CadreID).
		Set("Designation_ID", value1.DesignationID).
		Set("Post_Name", value1.PostName).
		Where(sq.And{
			sq.Eq{"post_id": value.PostID},
			sq.Eq{"office_id": value.RedeploymentFromOfficeID},
		})

	if err := dblib.QueueExecRow(batch, uquery); err != nil {
		return err
	}

	br := tx.SendBatch(gctx, batch)
	defer br.Close()

	// Check insert result
	if _, err := br.Exec(); err != nil {
		log.Error(gctx, "Insert query failed: %v", err)
		return err
	}

	// Check update result
	if _, err := br.Exec(); err != nil {
		log.Error(gctx, "Update query failed: %v", err)
		return err
	}
	// Final update
	if _, err := br.Exec(); err != nil {
		return fmt.Errorf("update current post query failed: %w", err)
	}
	// If this post is newly set as Head of Office, call external API
	if value1.IsHeadOfTheOffice {
		updateReq := UpdateHeadOfficeRequest{
			OfficeID:      value.RedeploymentToOfficeID,
			NewHeadPostID: value.PostID,
		}
		reqBody, err1 := json.Marshal(updateReq)
		if err1 != nil {
			return fmt.Errorf("marshal UpdateHeadOfficeRequest failed: %w", err1)
		}

		baseURL := rmr.cfg.GetString("url.masterdatabaseurl")
		if baseURL == "" {
			return fmt.Errorf("masterdatabaseurl is empty in config")
		}

		apiURL := fmt.Sprintf("%soffices/update-head-office", strings.TrimRight(baseURL, "/")+"/")
		req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(reqBody))
		if err != nil {
			return fmt.Errorf("new request for update-head-office failed: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		client := &http.Client{Timeout: 10 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			return fmt.Errorf("call to update-head-office API failed: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
			return fmt.Errorf("update-head-office API returned status: %s", resp.Status)
		}
	}
	return nil
}

func (r *PosttoPostMappingRepository) BeginTransaction(ctx context.Context) (pgx.Tx, error) {
	return r.db.Pool.Begin(ctx)
}

func (r *PosttoPostMappingRepository) CompensateFailedRedeployment(ctx context.Context, postID int, filePath string) error {
	// Example: Delete the redeployment log
	_, err := r.db.Exec(ctx, "DELETE FROM pmdm.post_redeployment_log WHERE post_id = $1 AND upload_order_doc_name = $2", postID, filePath)
	return err
}

func (rmr *PosttoPostMappingRepository) IsPostOfficeMappingValid(ctx context.Context, postID int, officeID int) (bool, error) {
	var exists bool
	query := `SELECT EXISTS (
        SELECT 1 FROM pmdm.post_management_master
        WHERE post_id = $1 AND office_id = $2
    )`
	err := rmr.db.Pool.QueryRow(ctx, query, postID, officeID).Scan(&exists)
	if err != nil {
		return false, err
	}
	return exists, nil
}
func (rmr *PosttoPostMappingRepository) UploadFile(gctx *gin.Context, miniofile *domain.MinioFile) error {
	_, cancel := context.WithTimeout(gctx.Request.Context(), rmr.cfg.GetDuration("db.QueryTimeoutLow"))
	defer cancel()

	return InsertFile(gctx, miniofile, rmr.minio, rmr.cfg.GetString("minio.bucketName"))
}
func InsertFile(ctx context.Context, file *domain.MinioFile, minioClient *minio.Client, bucketName string) error {
	log.Debug(ctx, "InsertFile started")

	_, err := minioClient.PutObject(
		ctx,
		bucketName,
		file.FilePath,
		file.File,
		file.FileSize,
		minio.PutObjectOptions{ContentType: file.ContentType},
	)

	if err != nil {
		log.Error(ctx, "Error in minioutil.InsertFile:%s", err.Error())
		return err
	}
	log.Debug(ctx, "PutObject completed")
	return nil
}

// func (rmr *PosttoPostMappingRepository) SavePostRedeploymentRepo(ctx *gin.Context, tx pgx.Tx, value domain.PostRedeploymentLog,
// 	value1 domain.PostManagementMasterUpdate) error {

// 	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
// 	defer cancel()

// 	batch := &pgx.Batch{}

// 	// Insert into post_redeployment_log
// 	query := dblib.Psql.Insert("pmdm.post_redeployment_log").
// 		Columns(
// 			"post_id",
// 			"cadre_name",
// 			"redeployment_from_office_id",
// 			"redeployment_to_office_id",
// 			"redeployment_on",
// 			"redeployment_by",
// 			"effective_from",
// 			"effective_upto",
// 		).
// 		Values(
// 			value.PostID,
// 			value.CadreName,
// 			value.RedeploymentFromOfficeID,
// 			value.RedeploymentToOfficeID,
// 			value.RedeploymentOn,
// 			value.RedeploymentBy,
// 			value.EffectiveFrom,
// 			value.EffectiveUPTo,
// 		)

// 	if err := dblib.QueueExecRow(batch, query); err != nil {
// 		return err
// 	}
// 	// Check and update existing head of office if value1.IsHeadOfTheOffice is true
// 	if value1.IsHeadOfTheOffice {
// 		// Check if another post has Is_Head_of_the_Office = true for same office
// 		row := tx.QueryRow(gctx, `
// 			SELECT post_id FROM pmdm.post_management_master
// 			WHERE office_id = $1 AND is_head_of_the_office = true`, value.RedeploymentToOfficeID)

// 		var prevHeadPostID int
// 		err := row.Scan(&prevHeadPostID)
// 		if err == nil {
// 			// Previous head exists, update is_head_of_the_office to false
// 			headUpdate := dblib.Psql.Update("pmdm.post_management_master").
// 				Set("Is_Head_of_the_Office", false).
// 				Where(sq.Eq{"post_id": prevHeadPostID})

// 			if err := dblib.QueueExecRow(batch, headUpdate); err != nil {
// 				return fmt.Errorf("failed to queue head office flag removal: %w", err)
// 			}
// 		} else if err != pgx.ErrNoRows {
// 			return fmt.Errorf("error checking existing head of office: %w", err)
// 		}
// 	}
// 	// Update post_management_master using value1 for all master fields
// 	uquery := dblib.Psql.Update("pmdm.post_management_master").
// 		Set("Office_id", value1.OfficeID).
// 		Set("Office_Name", value1.OfficeName).
// 		Set("Filled_Status", value1.FilledStatus).
// 		Set("Post_Status", value1.PostStatus).
// 		Set("Allowances_Attached", value1.AllowancesAttached).
// 		Set("Allowance_Description", value1.AllowanceDescription).
// 		Set("Updated_By", value1.UpdatedBy).
// 		Set("Updated_Date", value1.UpdatedDate).
// 		Set("Status", value1.Status).
// 		Set("Remarks", value1.Remarks).
// 		Set("Valid_From", value1.ValidFrom).
// 		Set("Valid_To", value1.ValidTo).
// 		Set("Order_Casemark", value1.OrderCasemark).
// 		Set("Order_Date", value1.OrderDate).
// 		Set("Establishment_Register_ID", value1.EstablishmentRegisterID).
// 		Set("Designation", value1.Designation).
// 		Set("Permanent_Status", value1.PermanentStatus).
// 		Set("Establishment_Register_Name", value1.EstablishmentRegisterName).
// 		Set("Group_Name", value1.GroupName).
// 		Set("Office_Type", value1.OfficeType).
// 		Set("Office_Supervisor", value1.OfficeSupervisor).
// 		Set("Is_Head_of_the_Office", value1.IsHeadOfTheOffice).
// 		Set("Cadre_ID", value1.CadreID).
// 		Set("Designation_ID", value1.DesignationID).
// 		Set("Post_Name", value1.PostName).
// 		Where(sq.And{
// 			sq.Eq{"post_id": value.PostID},
// 			sq.Eq{"office_id": value.RedeploymentFromOfficeID},
// 		})

// 	if err := dblib.QueueExecRow(batch, uquery); err != nil {
// 		return err
// 	}

// 	br := tx.SendBatch(gctx, batch)
// 	defer br.Close()

// 	// insert result
// 	if _, err := br.Exec(); err != nil {
// 		log.Error(gctx, "Insert query failed: %v", err)
// 		return err
// 	}

// 	// Conditionally execute if head-of-office update was queued
// 	if value1.IsHeadOfTheOffice {
// 		if _, err := br.Exec(); err != nil {
// 			log.Error(gctx, "Update query failed: %v", err)
// 			return err
// 		}
// 	}
// 	// Update post_management_master
// 	if _, err := br.Exec(); err != nil {
// 		return fmt.Errorf("update current post query failed: %w", err)
// 	}
// 	// If this post is newly set as Head of Office, call external API
// 	if value1.IsHeadOfTheOffice {
// 		updateReq := UpdateHeadOfficeRequest{
// 			OfficeID:      value.RedeploymentToOfficeID,
// 			NewHeadPostID: value.PostID,
// 		}
// 		reqBody, err1 := json.Marshal(updateReq)
// 		if err1 != nil {
// 			return fmt.Errorf("marshal UpdateHeadOfficeRequest failed: %w", err1)
// 		}

// 		baseURL := rmr.cfg.GetString("url.masterdatabaseurl")
// 		if baseURL == "" {
// 			return fmt.Errorf("masterdatabaseurl is empty in config")
// 		}

// 		apiURL := fmt.Sprintf("%soffices/update-head-office", strings.TrimRight(baseURL, "/")+"/")
// 		req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(reqBody))
// 		if err != nil {
// 			return fmt.Errorf("new request for update-head-office failed: %w", err)
// 		}
// 		req.Header.Set("Content-Type", "application/json")
// 		client := &http.Client{Timeout: 10 * time.Second}
// 		resp, err := client.Do(req)
// 		if err != nil {
// 			return fmt.Errorf("call to update-head-office API failed: %w", err)
// 		}
// 		defer resp.Body.Close()

// 		if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
// 			return fmt.Errorf("update-head-office API returned status: %s", resp.Status)
// 		}
// 	}
// 	return nil
// }

func (rmr *PosttoPostMappingRepository) SavePostRedeploymentRepo(
	ctx *gin.Context,
	tx pgx.Tx,
	value domain.PostRedeploymentLog,
	value1 domain.PostManagementMasterUpdate,
) error {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	batch := &pgx.Batch{}
	resultsQueued := 0

	// Insert into post_redeployment_log
	query := dblib.Psql.Insert("pmdm.post_redeployment_log").
		Columns(
			"post_id",
			"cadre_name",
			"redeployment_from_office_id",
			"redeployment_to_office_id",
			"redeployment_on",
			"redeployment_by",
			"effective_from",
			"effective_upto",
		).
		Values(
			value.PostID,
			value.CadreName,
			value.RedeploymentFromOfficeID,
			value.RedeploymentToOfficeID,
			value.RedeploymentOn,
			value.RedeploymentBy,
			value.EffectiveFrom,
			value.EffectiveUPTo,
		)

	if err := dblib.QueueExecRow(batch, query); err != nil {
		return err
	}
	resultsQueued++ // 1: insert

	// Check and update existing head of office if value1.IsHeadOfTheOffice is true
	if value1.IsHeadOfTheOffice {
		row := tx.QueryRow(gctx, `
			SELECT post_id FROM pmdm.post_management_master 
			WHERE office_id = $1 AND is_head_of_the_office = true`, value.RedeploymentToOfficeID)

		var prevHeadPostID int
		err := row.Scan(&prevHeadPostID)
		if err == nil {
			// Previous head exists, queue update is_head_of_the_office=false
			headUpdate := dblib.Psql.Update("pmdm.post_management_master").
				Set("Is_Head_of_the_Office", false).
				Where(sq.Eq{"post_id": prevHeadPostID})

			if err := dblib.QueueExecRow(batch, headUpdate); err != nil {
				return fmt.Errorf("failed to queue head office flag removal: %w", err)
			}
			resultsQueued++ // 2: head update
		} else if err != pgx.ErrNoRows {
			return fmt.Errorf("error checking existing head of office: %w", err)
		}
	}

	// Update post_management_master
	uquery := dblib.Psql.Update("pmdm.post_management_master").
		Set("Office_id", value1.OfficeID).
		Set("Office_Name", value1.OfficeName).
		Set("Filled_Status", value1.FilledStatus).
		Set("Post_Status", value1.PostStatus).
		Set("Allowances_Attached", value1.AllowancesAttached).
		Set("Allowance_Description", value1.AllowanceDescription).
		Set("Updated_By", value1.UpdatedBy).
		Set("Updated_Date", value1.UpdatedDate).
		Set("Status", value1.Status).
		Set("Remarks", value1.Remarks).
		Set("Valid_From", value1.ValidFrom).
		Set("Valid_To", value1.ValidTo).
		Set("Order_Casemark", value1.OrderCasemark).
		Set("Order_Date", value1.OrderDate).
		Set("Establishment_Register_ID", value1.EstablishmentRegisterID).
		Set("Designation", value1.Designation).
		Set("Permanent_Status", value1.PermanentStatus).
		Set("Establishment_Register_Name", value1.EstablishmentRegisterName).
		Set("Group_Name", value1.GroupName).
		Set("Office_Type", value1.OfficeType).
		Set("Office_Supervisor", value1.OfficeSupervisor).
		Set("Is_Head_of_the_Office", value1.IsHeadOfTheOffice).
		Set("Cadre_ID", value1.CadreID).
		Set("Designation_ID", value1.DesignationID).
		Set("Post_Name", value1.PostName).
		Where(sq.And{
			sq.Eq{"post_id": value.PostID},
			sq.Eq{"office_id": value.RedeploymentFromOfficeID},
		})

	if err := dblib.QueueExecRow(batch, uquery); err != nil {
		return err
	}
	resultsQueued++ // last: update

	// Send the batch
	br := tx.SendBatch(gctx, batch)
	defer br.Close()

	// Execute all queued statements
	for i := 0; i < resultsQueued; i++ {
		if _, err := br.Exec(); err != nil {
			return fmt.Errorf("batch exec %d failed: %w", i+1, err)
		}
	}

	// If this post is newly set as Head of Office, call external API
	if value1.IsHeadOfTheOffice {
		updateReq := UpdateHeadOfficeRequest{
			OfficeID:      value.RedeploymentToOfficeID,
			NewHeadPostID: value.PostID,
		}
		reqBody, err1 := json.Marshal(updateReq)
		if err1 != nil {
			return fmt.Errorf("marshal UpdateHeadOfficeRequest failed: %w", err1)
		}

		baseURL := rmr.cfg.GetString("url.masterdatabaseurl")
		if baseURL == "" {
			return fmt.Errorf("masterdatabaseurl is empty in config")
		}

		// apiURL := fmt.Sprintf("%soffices/update-head-office", strings.TrimRight(baseURL, "/"))
		apiURL := fmt.Sprintf("%s/%s", strings.TrimRight(baseURL, "/"), path.Join("offices", "update-head-office"))
		log.Info(ctx, "Calling update-head-office API: %s, payload: %s", apiURL, string(reqBody))
		req, err := http.NewRequest("POST", apiURL, bytes.NewBuffer(reqBody))
		if err != nil {
			return fmt.Errorf("new request for update-head-office failed: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		client := &http.Client{Timeout: 10 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			return fmt.Errorf("call to update-head-office API failed: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
			return fmt.Errorf("update-head-office API returned status: %s", resp.Status)
		}
	}
	return nil
}

type UpdateHeadOfficeRequest struct {
	OfficeID      int `json:"office_id" binding:"required"`
	NewHeadPostID int `json:"new_head_post_id" binding:"required"`
}

func (rmr *PosttoPostMappingRepository) UpdateRedeployedPostAuthorityCharges(ctx *gin.Context, postID int) (int, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := `
		UPDATE pmdm.post_mapping_detail
		SET 
			leave_sanc_authority_1 = CASE WHEN leave_sanc_authority_1 = $1 THEN NULL ELSE leave_sanc_authority_1 END,
			leave_sanc_authority_2 = CASE WHEN leave_sanc_authority_2 = $1 THEN NULL ELSE leave_sanc_authority_2 END,
			pay_approve_authority1 = CASE WHEN pay_approve_authority1 = $1 THEN NULL ELSE pay_approve_authority1 END,
			appointing_authority = CASE WHEN appointing_authority = $1 THEN NULL ELSE appointing_authority END,
			disciplinary_authority = CASE WHEN disciplinary_authority = $1 THEN NULL ELSE disciplinary_authority END,
			ddo_authority = CASE WHEN ddo_authority = $1 THEN NULL ELSE ddo_authority END
		WHERE 
			leave_sanc_authority_1 = $1 OR
			leave_sanc_authority_2 = $1 OR
			pay_approve_authority1 = $1 OR
			appointing_authority = $1 OR
			disciplinary_authority = $1 OR
			ddo_authority = $1;
	`

	tag, err := rmr.db.Exec(gctx, query, postID)
	if err != nil {
		return 0, fmt.Errorf("query execution error: %w", err)
	}

	rowsAffected := tag.RowsAffected()
	if rowsAffected == 0 {
		log.Error(ctx, "No rows updated for post_id: %d", postID)
	}

	return int(rowsAffected), nil
}

func (rmr *PosttoPostMappingRepository) GetRedeployedPostAuthorityCharges(ctx *gin.Context, post_id int64) ([]domain.RedeployedPostAuthority, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	// 1. Query for counts
	countsQuery := `
    WITH column_matches AS (
        SELECT 'leave_sanc_authority_1' AS authority_name, COUNT(*) AS match_count 
        FROM pmdm.post_mapping_detail WHERE leave_sanc_authority_1 = $1
        UNION ALL
        SELECT 'leave_sanc_authority_2', COUNT(*) 
        FROM pmdm.post_mapping_detail WHERE leave_sanc_authority_2 = $1
        UNION ALL
        SELECT 'pay_approve_authority1', COUNT(*) 
        FROM pmdm.post_mapping_detail WHERE pay_approve_authority1 = $1
        UNION ALL
        SELECT 'appointing_authority', COUNT(*) 
        FROM pmdm.post_mapping_detail WHERE appointing_authority = $1
        UNION ALL
        SELECT 'disciplinary_authority', COUNT(*) 
        FROM pmdm.post_mapping_detail WHERE disciplinary_authority = $1
        UNION ALL
        SELECT 'ddo_authority', COUNT(*) 
        FROM pmdm.post_mapping_detail WHERE ddo_authority = $1
    )
    SELECT authority_name, match_count
    FROM column_matches
    WHERE match_count > 0
    ORDER BY match_count DESC;`

	countsRows, err := rmr.db.Query(gctx, countsQuery, post_id)
	if err != nil {
		return nil, fmt.Errorf("error executing counts query: %w", err)
	}
	defer countsRows.Close()

	var results []domain.RedeployedPostAuthority
	for countsRows.Next() {
		var authorityName string
		var matchCount int

		if err := countsRows.Scan(&authorityName, &matchCount); err != nil {
			return nil, fmt.Errorf("error scanning counts row: %w", err)
		}

		results = append(results, domain.RedeployedPostAuthority{
			AuthorityName:   authorityName,
			EmployeeCount:   matchCount,
			EmployeeDetails: make([]domain.EmployeeDetail, 0),
		})
	}

	if len(results) == 0 {
		return results, nil
	}

	// 2. Prepare and execute detailed employees query
	detailQuery := `
    SELECT 
        CONCAT(
            kem.employee_first_name, 
            COALESCE(CONCAT(' ', kem.employee_middle_name), ''), 
            COALESCE(CONCAT(' ', kem.employee_last_name), '')
        ) AS employee_name,
        kem.employee_id,
        km.post_id,
        km.post_name,
        om.office_id,
        om.office_name,
        pmd.leave_sanc_authority_1,
        pmd.leave_sanc_authority_2,
        pmd.pay_approve_authority1,
        pmd.appointing_authority,
        pmd.disciplinary_authority,
        pmd.ddo_authority
    FROM 
        pmdm.post_mapping_detail pmd
    LEFT JOIN 
        pmdm.kafka_employee_master kem ON kem.post_id = pmd.employee_post_id
    LEFT JOIN 
        pmdm.post_management_master km ON km.post_id = pmd.employee_post_id
    LEFT JOIN 
        pmdm.kafka_office_master om ON om.office_id = kem.office_id
    WHERE 
        $1 IN (
            pmd.employee_post_id,
            pmd.leave_sanc_authority_1,
            pmd.leave_sanc_authority_2,
            pmd.pay_approve_authority1,
            pmd.appointing_authority,
            pmd.disciplinary_authority,
            pmd.ddo_authority
        );`

	detailRows, err := rmr.db.Query(gctx, detailQuery, post_id)
	if err != nil {
		return nil, fmt.Errorf("error executing details query: %w", err)
	}
	defer detailRows.Close()

	for detailRows.Next() {
		var ed domain.EmployeeDetail
		var leaveSanc1, leaveSanc2, payApprove1, appointing, disciplinary, ddo sql.NullInt64

		if err := detailRows.Scan(
			&ed.EmployeeName,
			&ed.EmployeeID,
			&ed.PostID,
			&ed.PostName,
			&ed.OfficeID,
			&ed.OfficeName,
			&leaveSanc1,
			&leaveSanc2,
			&payApprove1,
			&appointing,
			&disciplinary,
			&ddo,
		); err != nil {
			return nil, fmt.Errorf("error scanning detail row: %w", err)
		}

		// Assign employee to matching authorities in results
		for i := range results {
			switch results[i].AuthorityName {
			case "leave_sanc_authority_1":
				if leaveSanc1.Valid && leaveSanc1.Int64 == post_id {
					results[i].EmployeeDetails = append(results[i].EmployeeDetails, ed)
				}
			case "leave_sanc_authority_2":
				if leaveSanc2.Valid && leaveSanc2.Int64 == post_id {
					results[i].EmployeeDetails = append(results[i].EmployeeDetails, ed)
				}
			case "pay_approve_authority1":
				if payApprove1.Valid && payApprove1.Int64 == post_id {
					results[i].EmployeeDetails = append(results[i].EmployeeDetails, ed)
				}
			case "appointing_authority":
				if appointing.Valid && appointing.Int64 == post_id {
					results[i].EmployeeDetails = append(results[i].EmployeeDetails, ed)
				}
			case "disciplinary_authority":
				if disciplinary.Valid && disciplinary.Int64 == post_id {
					results[i].EmployeeDetails = append(results[i].EmployeeDetails, ed)
				}
			case "ddo_authority":
				if ddo.Valid && ddo.Int64 == post_id {
					results[i].EmployeeDetails = append(results[i].EmployeeDetails, ed)
				}
			}
		}
	}
	return results, nil
}
