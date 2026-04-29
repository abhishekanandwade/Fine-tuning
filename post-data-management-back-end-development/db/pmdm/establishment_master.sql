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