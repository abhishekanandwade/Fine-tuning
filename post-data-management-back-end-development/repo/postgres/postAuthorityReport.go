package repo

import (
	"context"
	"time"

	"pmdm/core/domain"

	sq "github.com/Masterminds/squirrel"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5"
	dblib "gitlab.cept.gov.in/it-2.0-common/api-db"
)

func (rmr *PostManagementRepository) GetPostAuthorityChargesDetailByOfficeRepo(ctx *gin.Context, officeID int64) ([]domain.PostAuthorityDetails, error) {
	gctx, cancel := context.WithTimeout(ctx.Request.Context(), 10*time.Second)
	defer cancel()

	query := dblib.Psql.Select(
		"pmm.post_id",
		"pmm.post_name",
		"pmm.designation",
		"pmm.cadre_id",
		"cm.cadre_name",
		"pmm.group_id",
		"gm.group_name",
		"kem.employee_id",
		"TRIM(CONCAT_WS(' ', kem.employee_first_name, kem.employee_middle_name, kem.employee_last_name)) AS employee_name",
		"pmd.leave_sanc_authority_1",
		"pmd.leave_sanc_authority_2",
		"pmd.pay_approve_authority1",
		"pmd.appointing_authority",
		"pmd.disciplinary_authority",
		"pmd.ddo_authority",
	).From("pmdm.post_management_master pmm").
		LeftJoin("pmdm.kafka_office_master kom ON kom.office_id = pmm.office_id").
		LeftJoin("pmdm.post_mapping_detail pmd ON pmd.employee_post_id = pmm.post_id").
		LeftJoin("pmdm.kafka_employee_master kem ON kem.post_id = pmm.post_id").
		LeftJoin("pmdm.cadre_master cm ON cm.cadre_id = pmm.cadre_id").
		LeftJoin("pmdm.group_master gm ON gm.group_id = pmm.group_id").
		Where(sq.Eq{
			"pmm.office_id": officeID,
			"pmm.status":    []string{"Active", "Abeyance", "Surplus"},
		})
	return dblib.SelectRows(gctx, rmr.db, query, pgx.RowToStructByNameLax[domain.PostAuthorityDetails])
}
