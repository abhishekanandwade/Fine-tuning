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