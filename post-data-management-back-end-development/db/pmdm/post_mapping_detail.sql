-- pmdm.post_mapping_detail definition

-- Drop table

-- DROP TABLE pmdm.post_mapping_detail;

CREATE TABLE pmdm.post_mapping_detail (
	employee_post_id int4 NOT NULL,
	gds_leave_sanc_authority_1 int4 NULL,
	gds_leave_sanc_authority_2 int4 NULL,
	reporting_authority int4 NULL,
	apar_reporting_authority int4 NULL,
	apar_review_authority int4 NULL,
	apar_accepting_authority int4 NULL,
	apar_represent_authority int4 NULL,
	service_book_approve_authority1 int4 NULL,
	leave_sanc_authority_1 int4 NULL,
	leave_sanc_authority_2 int4 NULL,
	leave_sanc_authority_3 int4 NULL,
	updated_date timestamp NULL,
	updated_by varchar(30) NULL,
	pay_approve_authority1 int4 NULL,
	pay_approve_authority2 int4 NULL,
	leave_fwd_authority1 int4 NULL,
	leave_fwd_authority2 int4 NULL,
	pay_fwd_authority1 int4 NULL,
	pay_fwd_authority2 int4 NULL,
	appointing_authority int4 NULL,
	disciplinary_authority int4 NULL,
	ddo_authority int4 NULL,
	admin_authority int4 NULL,
	pension_sanctioning_authority int4 NULL,
	pension_authorising_authority int4 NULL,
	service_book_approve_authority2 int4 NULL,
	post_map_id serial4 NOT NULL,
	role_authority int4 NULL,
	employee_office_id int4 NULL,
	service_book_foward_authority1 int4 NULL,
	"post-map_id1" serial4 NOT NULL,
	service_book_foward_authority2 int4 NULL,
	vigilence_maker_authority int4 NULL,
	admin_office int4 NULL,
	CONSTRAINT post_mapping_detail_pk PRIMARY KEY (post_map_id),
	CONSTRAINT post_mapping_detail_un UNIQUE (employee_post_id)
);