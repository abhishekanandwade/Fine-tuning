-- DROP SCHEMA pmdm;

CREATE SCHEMA pmdm AUTHORIZATION pmdm_admin;

-- DROP SEQUENCE pmdm.cadre_mastser_cadre_code_seq;

CREATE SEQUENCE pmdm.cadre_mastser_cadre_code_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE pmdm.designation_master_designation_id_seq;

CREATE SEQUENCE pmdm.designation_master_designation_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE pmdm.document_master_pmdm_document_id_seq;

CREATE SEQUENCE pmdm.document_master_pmdm_document_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE pmdm.document_master_pmdm_post_id_seq;

CREATE SEQUENCE pmdm.document_master_pmdm_post_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE pmdm.establishment_master_establishment_register_id_seq;

CREATE SEQUENCE pmdm.establishment_master_establishment_register_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE pmdm.post_mapping_detail_maker1_post_map_id_seq;

CREATE SEQUENCE pmdm.post_mapping_detail_maker1_post_map_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE pmdm.post_mapping_detail_maker1_post_mapping_detail_maker1_id_seq;

CREATE SEQUENCE pmdm.post_mapping_detail_maker1_post_mapping_detail_maker1_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE pmdm.post_mapping_detail_maker_log_post_map_id_seq;

CREATE SEQUENCE pmdm.post_mapping_detail_maker_log_post_map_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE pmdm.post_mapping_detail_maker_log_post_mapping_detail_maker_id_seq;

CREATE SEQUENCE pmdm.post_mapping_detail_maker_log_post_mapping_detail_maker_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE pmdm.post_mapping_detail_maker_post_map_id_seq;

CREATE SEQUENCE pmdm.post_mapping_detail_maker_post_map_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE pmdm.post_mapping_detail_maker_post_mapping_detail_maker_id_seq;

CREATE SEQUENCE pmdm.post_mapping_detail_maker_post_mapping_detail_maker_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE pmdm."post_mapping_detail_post-map_id1_seq";

CREATE SEQUENCE pmdm."post_mapping_detail_post-map_id1_seq"
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE pmdm.post_mapping_detail_post_map_id_seq;

CREATE SEQUENCE pmdm.post_mapping_detail_post_map_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE pmdm.postmanagement_master_maker_postmanagement_maker_id_seq;

CREATE SEQUENCE pmdm.postmanagement_master_maker_postmanagement_maker_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE pmdm.postmanagement_master_post_id1_seq;

CREATE SEQUENCE pmdm.postmanagement_master_post_id1_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;
-- DROP SEQUENCE pmdm.postmanagement_master_postmanagement_id_seq;

CREATE SEQUENCE pmdm.postmanagement_master_postmanagement_id_seq
	INCREMENT BY 1
	MINVALUE 1
	MAXVALUE 2147483647
	START 1
	CACHE 1
	NO CYCLE;-- pmdm.cadre_master definition

-- Drop table

-- DROP TABLE pmdm.cadre_master;

CREATE TABLE pmdm.cadre_master (
	cadre_id int4 DEFAULT nextval('pmdm.cadre_mastser_cadre_code_seq'::regclass) NOT NULL,
	cadre_name varchar(50) NULL,
	group_name varchar(30) NULL,
	pay_level int4 NULL,
	grade_pay int4 NULL,
	created_by varchar(30) NULL,
	created_date timestamp NULL,
	approved_by varchar(30) NULL,
	approved_date timestamp NULL,
	updated_by varchar(30) NULL,
	updated_date timestamp NULL,
	valid_from timestamp NULL,
	valid_to timestamp NULL,
	status varchar(50) NULL,
	remarks varchar(200) NULL,
	group_id int4 NULL,
	CONSTRAINT cadre_mastser_pk PRIMARY KEY (cadre_id)
);


-- pmdm.designation_master definition

-- Drop table

-- DROP TABLE pmdm.designation_master;

CREATE TABLE pmdm.designation_master (
	designation_id serial4 NOT NULL,
	designation varchar(50) NULL,
	group_name varchar(30) NULL,
	cadre_name varchar(50) NULL,
	created_by varchar(30) NULL,
	created_date timestamp NULL,
	approved_by varchar(30) NULL,
	approved_date timestamp NULL,
	valid_from timestamp NULL,
	valid_to timestamp NULL,
	status varchar(30) NULL,
	remarks varchar(200) NULL,
	cadre_id int4 NULL,
	group_id int4 NULL,
	CONSTRAINT designation_master_pk PRIMARY KEY (designation_id)
);


-- pmdm.document_master_pmdm definition

-- Drop table

-- DROP TABLE pmdm.document_master_pmdm;

CREATE TABLE pmdm.document_master_pmdm (
	post_id serial4 NOT NULL,
	order_casemark varchar(50) NULL,
	order_date timestamp NULL,
	document_name varchar(50) NOT NULL,
	document_type varchar(30) NOT NULL,
	document_size int4 NOT NULL,
	document_approver_post_id varchar(30) NULL,
	document_upload_status varchar(30) NOT NULL,
	document_uploaded_by varchar(30) NULL,
	document_uploaded_date timestamp NULL,
	document_updated_by varchar(30) NULL,
	document_updated_date timestamp NULL,
	document_approved_by varchar(30) NULL,
	document_approved_date timestamp NULL,
	remarks varchar(200) NULL,
	document_file_path varchar(200) NULL,
	office_id int4 NULL,
	document_id serial4 NOT NULL,
	CONSTRAINT document_master_pmdm_pk PRIMARY KEY (document_id)
);


-- pmdm.establishment_master definition

-- Drop table

-- DROP TABLE pmdm.establishment_master;

CREATE TABLE pmdm.establishment_master (
	establishment_register_id serial4 NOT NULL,
	office_id int4 NOT NULL,
	office_name varchar(50) NULL,
	office_type varchar(30) NULL,
	sanctioned_strength int4 NULL,
	groupa_posts_count int4 NULL,
	groupb_gazitted_posts_count int4 NULL,
	groupb_nongazitted_posts_count int4 NULL,
	groupc_posts_count int4 NULL,
	approve_authority_postid varchar(30) NULL,
	establishment_order_docname varchar(50) NULL,
	created_by varchar(30) NULL,
	created_date timestamp NULL,
	updated_by varchar(30) NULL,
	updated_date timestamp NULL,
	approved_by varchar(30) NULL,
	approved_date timestamp NULL,
	status varchar(30) NULL,
	remarks varchar(200) NULL,
	circle_id varchar(30) NULL,
	circle_name varchar(50) NULL,
	region_id varchar(30) NULL,
	region_name varchar(50) NULL,
	division_id varchar(30) NULL,
	division_name varchar(50) NULL,
	ho_id varchar(30) NULL,
	ho_name varchar(50) NULL,
	so_id varchar(30) NULL,
	so_name varchar(50) NULL,
	rms_status bool NULL,
	establishment_register_name varchar(50) NULL,
	gds_posts_count int4 NULL,
	reporting_office_id int4 NULL,
	reporting_office_name varchar(30) NULL,
	estt_register_number int4 NULL,
	CONSTRAINT establishment_master_pk PRIMARY KEY (establishment_register_id),
	CONSTRAINT establishment_master_un UNIQUE (office_id)
);


-- pmdm.kafka_office_hierarchy definition

-- Drop table

-- DROP TABLE pmdm.kafka_office_hierarchy;

CREATE TABLE pmdm.kafka_office_hierarchy (
	office_hierarchy_id int4 DEFAULT 0 NOT NULL,
	office_id int4 NOT NULL,
	office_type_id int4 NULL,
	office_type_code varchar(20) NULL,
	circle_name varchar(50) NULL,
	circle_code varchar(20) NULL,
	circle_office_id int4 NULL,
	region_name varchar(50) NULL,
	region_office_id int4 NULL,
	division_name varchar(50) NULL,
	division_office_id int4 NULL,
	subdivision_name varchar(50) NULL,
	subdivision_office_id int4 NULL,
	ho_id int4 NULL,
	ho_name varchar(50) NULL,
	hro_id int4 NULL,
	hro_name varchar(50) NULL,
	so_id int4 NULL,
	so_name varchar(50) NULL,
	sro_id int4 NULL,
	sro_name varchar(50) NULL,
	accounting_office_id int4 NULL,
	bo_id int4 DEFAULT 0 NULL,
	bo_name varchar(50) DEFAULT ''::text NULL,
	sub_division_name text DEFAULT ''::text NULL,
	sub_division_office_id int4 DEFAULT 0 NULL,
	CONSTRAINT kafka_office_hierarchy_pk PRIMARY KEY (office_id)
);


-- pmdm.kafka_office_master definition

-- Drop table

-- DROP TABLE pmdm.kafka_office_master;

CREATE TABLE pmdm.kafka_office_master (
	office_id int4 NOT NULL,
	office_name varchar(50) NULL,
	office_type_id int4 NULL,
	office_type_code varchar(20) NULL,
	email_id varchar(50) NULL,
	contact_number varchar(20) NULL,
	office_class varchar(50) NULL,
	pincode int4 NULL,
	reporting_office_id int4 NULL,
	office_status_id int4 NULL,
	csi_facility_id varchar(20) NULL,
	sol_id varchar(20) NULL,
	ddo_code varchar(20) NULL,
	office_level varchar(20) NULL,
	working_days varchar(100) NULL,
	accounting_office_id int4 DEFAULT 0 NULL,
	delivery_office_flag bool NULL,
	pli_id varchar(20) NULL,
	remarks varchar(200) NULL,
	valid_from date NULL,
	valid_to date NULL,
	CONSTRAINT kafka_office_master_pkey PRIMARY KEY (office_id)
);


-- pmdm.master_authorities definition

-- Drop table

-- DROP TABLE pmdm.master_authorities;

CREATE TABLE pmdm.master_authorities (
	office_id int4 NULL,
	office_type varchar(50) NULL,
	office_name varchar(50) NULL,
	cadre_name varchar(50) NULL,
	designation varchar(50) NULL,
	post_id int4 NULL,
	emp_id int4 NULL,
	role_mapping_id varchar(50) NULL,
	authority_description varchar(50) NULL,
	division_office_id int4 NULL,
	region_office_id int4 NULL,
	circle_office_id int4 NULL,
	admin_office_id int4 NULL
);


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


-- pmdm.post_mapping_detail_maker definition

-- Drop table

-- DROP TABLE pmdm.post_mapping_detail_maker;

CREATE TABLE pmdm.post_mapping_detail_maker (
	post_mapping_detail_maker_id serial4 NOT NULL,
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
	service_book_foward_authority2 int4 NULL,
	vigilence_maker_authority int4 NULL,
	admin_office int4 NULL,
	approve_status varchar(50) NULL,
	approve_post_id varchar(50) NULL,
	created_by varchar(50) NULL,
	created_date timestamp NULL,
	approved_by varchar(30) NULL,
	approved_date timestamp NULL,
	remarks varchar NULL,
	CONSTRAINT post_mapping_detail_maker_pk PRIMARY KEY (post_mapping_detail_maker_id)
);


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


-- pmdm.post_mapping_detail_maker_log definition

-- Drop table

-- DROP TABLE pmdm.post_mapping_detail_maker_log;

CREATE TABLE pmdm.post_mapping_detail_maker_log (
	post_mapping_detail_maker_id serial4 NOT NULL,
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
	service_book_foward_authority2 int4 NULL,
	vigilence_maker_authority int4 NULL,
	admin_office int4 NULL,
	approve_status varchar(50) NULL,
	approve_post_id varchar(50) NULL,
	created_by varchar(50) NULL,
	created_date timestamp NULL,
	approved_by varchar(30) NULL,
	approved_date timestamp NULL,
	remarks varchar NULL,
	CONSTRAINT post_mapping_detail_maker_log_pk PRIMARY KEY (post_mapping_detail_maker_id)
);


-- pmdm.post_mapping_master definition

-- Drop table

-- DROP TABLE pmdm.post_mapping_master;

CREATE TABLE pmdm.post_mapping_master (
	mapping_id varchar(30) NULL,
	post_mapping_column_name varchar(100) NULL,
	post_mapping_status varchar(30) NULL,
	remarks varchar(200) NULL,
	post_mapping_id int4 NOT NULL,
	post_mapping_description varchar(200) NULL,
	CONSTRAINT post_mapping_master_pk PRIMARY KEY (post_mapping_id),
	CONSTRAINT post_mapping_master_un UNIQUE (mapping_id)
);


-- pmdm.post_name_master definition

-- Drop table

-- DROP TABLE pmdm.post_name_master;

CREATE TABLE pmdm.post_name_master (
	post_name_id int4 NULL,
	post_name varchar(100) NULL,
	group_id int4 NULL,
	"group" varchar(50) NULL,
	cadre_id int4 NULL,
	cadre varchar(50) NULL
);


-- pmdm.postmanagement_master definition

-- Drop table

-- DROP TABLE pmdm.postmanagement_master;

CREATE TABLE pmdm.postmanagement_master (
	postmanagement_id serial4 NOT NULL,
	office_id int4 NOT NULL,
	post_name varchar(50) NULL,
	office_name varchar(50) NULL,
	group_id int4 NULL,
	cadre_name varchar(50) NULL,
	filled_status varchar(30) NULL,
	post_status varchar(30) NULL,
	allowances_attached bool DEFAULT false NULL,
	allowance_description varchar(100) NULL,
	created_by varchar(50) NULL,
	created_date timestamp NULL,
	approved_by varchar(30) NULL,
	approved_date timestamp NULL,
	updated_by varchar(30) NULL,
	updated_date timestamp NULL,
	status varchar(30) NULL,
	remarks varchar(200) NULL,
	valid_from timestamp NULL,
	valid_to timestamp NULL,
	order_casemark varchar(100) NULL,
	order_date timestamp NULL,
	upload_order_doc_name varchar(50) NULL,
	establishment_register_id int4 NULL,
	designation varchar(50) NULL,
	pay_level int4 NULL,
	grade_pay int4 NULL,
	permanent_status bool DEFAULT true NULL,
	establishment_register_name varchar(200) NULL,
	post_id int4 DEFAULT nextval('pmdm.postmanagement_master_post_id1_seq'::regclass) NOT NULL,
	employee_group varchar(30) NULL,
	sanctioned_strength int4 NULL,
	group_name varchar(50) NULL,
	cadre_id int4 NULL,
	designation_id int4 NULL,
	approve_post_id varchar(30) NULL,
	master_maker_id varchar(200) NULL,
	admin_office_id int4 NULL,
	employee_type varchar(30) NULL,
	office_type varchar(10) NULL,
	login_id int4 NULL,
	CONSTRAINT postmanagement_master_pk PRIMARY KEY (postmanagement_id),
	CONSTRAINT postmanagement_master_un UNIQUE (post_id)
);


-- pmdm.postmanagement_master_maker definition

-- Drop table

-- DROP TABLE pmdm.postmanagement_master_maker;

CREATE TABLE pmdm.postmanagement_master_maker (
	postmanagement_maker_id serial4 NOT NULL,
	office_id int4 NOT NULL,
	post_name varchar(50) NULL,
	office_name varchar(50) NULL,
	group_id int4 NULL,
	cadre_name varchar(50) NULL,
	filled_status varchar(30) NULL,
	post_status varchar(30) NULL,
	allowances_attached bool DEFAULT false NULL,
	allowance_description varchar(100) NULL,
	created_by varchar(50) NULL,
	created_date timestamp NULL,
	approved_by varchar(30) NULL,
	approved_date timestamp NULL,
	updated_by varchar(30) NULL,
	updated_date timestamp NULL,
	status varchar(30) NULL,
	remarks varchar(200) NULL,
	valid_from timestamp NULL,
	valid_to timestamp NULL,
	order_casemark varchar(100) NULL,
	order_date timestamp NULL,
	upload_order_doc_name varchar(50) NULL,
	establishment_register_id int4 NULL,
	designation varchar(50) NULL,
	pay_level int4 NULL,
	grade_pay int4 NULL,
	permanent_status bool DEFAULT true NULL,
	establishment_register_name varchar(200) NULL,
	post_id int4 NOT NULL,
	employee_group varchar(30) NULL,
	sanctioned_strength int4 NULL,
	cadre_id int4 NULL,
	designation_id int4 NULL,
	approve_status varchar(50) NULL,
	approve_post_id varchar(50) NULL,
	new_office_id int4 NULL,
	new_office_name varchar(50) NULL,
	exchange_post_id int4 NULL,
	master_maker_id varchar(200) NULL,
	CONSTRAINT postmanagement_master_maker_pk PRIMARY KEY (postmanagement_maker_id)
);


-- pmdm.role_authority_mapping definition

-- Drop table

-- DROP TABLE pmdm.role_authority_mapping;

CREATE TABLE pmdm.role_authority_mapping (
	reporting_authority int4 NULL,
	post_name varchar(50) NULL,
	new_office_id int4 NULL,
	office_name varchar(50) NULL,
	group_name varchar(50) NULL,
	cadre varchar(50) NULL,
	designation varchar(50) NULL,
	office_type_code varchar(50) NULL,
	circle_office_id int4 NULL,
	circle_name varchar(50) NULL,
	region_office_id int4 NULL,
	region_name varchar(50) NULL,
	division_office_id int4 NULL,
	division_name varchar(50) NULL,
	sub_division_office_id varchar(50) NULL,
	sub_division_name varchar(50) NULL,
	ho_id int4 NULL,
	ho_name varchar(50) NULL,
	so_id varchar(50) NULL,
	so_name varchar(50) NULL,
	hro_id varchar(50) NULL,
	hro_name varchar(50) NULL,
	sro_id varchar(50) NULL,
	sro_name varchar(50) NULL,
	bo_id varchar(50) NULL,
	bo_name varchar(50) NULL
);



-- DROP PROCEDURE pmdm.edit_table_schema();

CREATE OR REPLACE PROCEDURE pmdm.edit_table_schema()
 LANGUAGE plpgsql
AS $procedure$
DECLARE 
    v_destination_table_name TEXT := 'kafka_office_master'; 
    v_destination_mapping_schema TEXT := 'pmdm';
    v_columns_data TEXT[] := ARRAY[    	
'office_id int4',
'office_name varchar(50)',
'office_type_id int4',
'office_type_code varchar(20)',
'email_id varchar(50)',
'contact_number varchar(20)',
'office_class varchar(50)',

'reporting_office_id int4',
'office_status_id int4',

'closed_date timestamp',
'supported_document_path varchar(50)',
'admin_flag bool',
'delivery_office_flag bool',

'pli_id varchar(20)',
'gstn_code varchar(20)',
'pao_code varchar(20)',
'atm_id varchar(20)',
'qr_terminal_id varchar(20)',
'weg_code varchar(20)',
'ddo_code varchar(20)',
'office_level varchar(20)',
'dac varchar(20)',
'working_days varchar(100)',
'working_hours_from varchar(10)',
'working_hours_to varchar(10)',
'valid_from date',
'valid_to date',
'remarks varchar(200)',
'approval_status varchar(20)',
'approved_by varchar(50)',
'created_by varchar(50)',
'created_date timestamp',
'updated_by varchar(50)',
'updated_date timestamp',
'approved_date timestamp',
'modified_flag bool',
'deleted_flag bool',
'office_status varchar(50)',
'accounting_office_id int4',
'order_memo_number varchar(100)',
'admin_office_id int4',
'new_office_id int4',
'facility_id varchar'
    ];
    v_column_data TEXT;
    v_existing_data_type TEXT;
    v_sql TEXT;
BEGIN
    FOREACH v_column_data IN ARRAY v_columns_data 
    LOOP
        -- Check if the column exists in the kafka_authrority_mapping table
		--RAISE NOTICE 'SELECT data_type FROM information_schema.columns WHERE table_schema = % AND table_name = % AND column_name = %', v_destination_mapping_schema, v_destination_table_name, split_part(v_column_data, ' ', 1);
        EXECUTE format('SELECT data_type FROM information_schema.columns WHERE table_schema = %L AND table_name = %L AND column_name = %L', v_destination_mapping_schema, v_destination_table_name, split_part(v_column_data, ' ', 1))
        INTO v_existing_data_type;
        
		RAISE NOTICE 'Value: % %', v_existing_data_type,split_part(v_column_data, ' ', 2);
        -- If the column exists and the data types do not match, alter the data type in kafka_authrority_mapping
        IF v_existing_data_type<>'<NULL>' AND v_existing_data_type <> split_part(v_column_data, ' ', 2) THEN
			RAISE NOTICE 'ALTER TABLE %.% ALTER COLUMN % TYPE %', v_destination_mapping_schema, v_destination_table_name, split_part(v_column_data, ' ', 1), split_part(v_column_data, ' ', 2);
            v_sql := format('ALTER TABLE %s.%s ALTER COLUMN %s TYPE %s', v_destination_mapping_schema, v_destination_table_name, split_part(v_column_data, ' ', 1), split_part(v_column_data, ' ', 2));
            EXECUTE v_sql;
        END IF;
    END LOOP;
END;
$procedure$
;