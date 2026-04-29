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