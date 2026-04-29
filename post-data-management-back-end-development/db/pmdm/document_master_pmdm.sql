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