-- pmdm.post_mapping_detail_maker1 definition

-- Drop table

-- DROP TABLE pmdm.post_mapping_detail_maker1;

CREATE TABLE pmdm.post_mapping_detail_maker1 (
	post_mapping_detail_maker1_id serial4 NOT NULL,
	employee_post_id int4 NOT NULL,
	updated_date timestamp NULL,
	updated_by varchar(30) NULL,
	post_map_id serial4 NOT NULL,
	employee_office_id int4 NULL,
	approve_status varchar(50) NULL,
	approve_post_id varchar(50) NULL,
	created_by varchar(50) NULL,
	created_date timestamp NULL,
	approved_by varchar(30) NULL,
	approved_date timestamp NULL,
	field_name varchar(100) NULL,
	field_value int4 NULL,
	CONSTRAINT post_mapping_detail_maker1_pk PRIMARY KEY (post_mapping_detail_maker1_id)
);