package repo

import (
	"context"
	"pmdm/core/domain"

	sq "github.com/Masterminds/squirrel"
	"github.com/gin-gonic/gin"
	"github.com/jackc/pgx/v5"
	dblib "gitlab.cept.gov.in/it-2.0-common/api-db"
)

func (pmr *PostManagementRepository) GetOfficeTypeByID(gctx *gin.Context, officeID int) (string, error) {
	ctx, cancel := context.WithTimeout(gctx.Request.Context(), pmr.cfg.GetDuration(DBtimeout))
	defer cancel()

	query := dblib.Psql.
		Select("COALESCE(office_type_code, '')").
		From("pmdm.kafka_office_hierarchy").
		Where(sq.Eq{"office_id": officeID})

	sql, args, err := query.ToSql()
	if err != nil {
		return "", err
	}

	var officeType string
	err = pmr.db.QueryRow(ctx, sql, args...).Scan(&officeType)
	if err != nil {
		return "", err
	}

	return officeType, nil
}

func (pmr *PostManagementRepository) FetchAllPostsByDivisionOfficeID(gctx *gin.Context, officeID int) ([]domain.PostManagementMasterNew, error) {
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
		Join("pmdm.kafka_office_hierarchy koh ON koh.office_id = pm.office_id").
		LeftJoin("pmdm.kafka_employee_master km ON km.post_id = pm.post_id AND km.employment_status = 'Active'").
		LeftJoin("pmdm.post_mapping_detail pmd ON pmd.employee_post_id = pm.post_id").
		LeftJoin("pmdm.kafka_office_master kom ON kom.office_id = pm.office_id").
		Where(sq.Eq{"koh.division_office_id": officeID})

	return dblib.SelectRows(ctx, pmr.db, query, pgx.RowToStructByNameLax[domain.PostManagementMasterNew])
}
