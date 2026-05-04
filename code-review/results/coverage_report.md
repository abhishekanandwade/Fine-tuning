# Agentic-RAG Coverage Report

- Repo:   `C:\Users\knchaitr\OneDrive - Hewlett Packard Enterprise\CoE Team\Fine-tuning\post-data-management-back-end-development`
- Report: `C:\Users\knchaitr\OneDrive - Hewlett Packard Enterprise\CoE Team\Fine-tuning\code-review\results\architectural_review.json`

## Summary

| Rule | Candidates | Confirmed | Missed | Extra | Recall % |
|---|---:|---:|---:|---:|---:|
| REPO-001 | 36 | 28 | 8 | 0 | 77.78 |
| REPO-002 | 17 | 15 | 2 | 0 | 88.24 |
| HANDLER-001 | 22 | 21 | 1 | 0 | 95.45 |
| OVERALL | 75 | 64 | 11 | 0 | 85.33 |

## Missed by agent — REPO-001 (8)

| File | Function | Start | End | Hits |
|---|---|---|---|---|
| repo/postgres/cadreMaster.go | ListCadresQry | 88 | 101 | 2 |
| repo/postgres/cadreMaster.go | UpdateCadreMasterQuery | 127 | 144 | 2 |
| repo/postgres/designationMaster.go | ListDesignationsQry | 48 | 59 | 2 |
| repo/postgres/designationMaster.go | UpdateDesignationMasterQuery | 154 | 171 | 2 |
| repo/postgres/postManagementMaster.go | GetDocumentsByOfficeID | 833 | 852 | 2 |
| repo/postgres/postManagementMaster.go | RejectPostManagementMaker | 1010 | 1051 | 3 |
| repo/postgres/postManagementMaster.go | CreatePostManagementMasterQuery | 1676 | 1878 | 2 |
| repo/postgres/postManagementMaster.go | CreatePost | 2103 | 2183 | 2 |

### Sample evidence

- **repo/postgres/cadreMaster.go::ListCadresQry** (lines 88-101)
  - line 90: ctx, cancel := context.WithTimeout(gctx.Request.Context(), cmr.cfg.GetDuration("db.QueryTimeoutMed"))
  - line 100: return dblib.SelectRows(ctx, cmr.db, query, pgx.RowToStructByNameLax[domain.CadreMaster])
- **repo/postgres/cadreMaster.go::UpdateCadreMasterQuery** (lines 127-144)
  - line 128: ctx, cancel := context.WithTimeout(gctx.Request.Context(), cmr.cfg.GetDuration("db.QueryTimeoutMed"))
  - line 138: resp, err := dblib.UpdateReturning(ctx, cmr.db, queryUpdate, pgx.RowToStructByNameLax[domain.CadreMaster])
- **repo/postgres/designationMaster.go::ListDesignationsQry** (lines 48-59)
  - line 50: ctx, cancel := context.WithTimeout(gctx.Request.Context(), dmr.cfg.GetDuration("db.QueryTimeoutMed"))
  - line 58: return dblib.SelectRows(ctx, dmr.db, query, pgx.RowToStructByNameLax[domain.DesignationMaster])
- **repo/postgres/designationMaster.go::UpdateDesignationMasterQuery** (lines 154-171)
  - line 155: ctx, cancel := context.WithTimeout(gctx.Request.Context(), dmr.cfg.GetDuration("db.QueryTimeoutMed"))
  - line 165: resp, err := dblib.UpdateReturning(ctx, dmr.db, queryUpdate, pgx.RowToStructByNameLax[domain.DesignationMaster])
- **repo/postgres/postManagementMaster.go::GetDocumentsByOfficeID** (lines 833-852)
  - line 835: query := psql.Select(
  - line 840: documents, err := dblib.SelectRows(ctx, r.db, query, pgx.RowToStructByNameLax[domain.Document])
- **repo/postgres/postManagementMaster.go::RejectPostManagementMaker** (lines 1010-1051)
  - line 1026: batch.Queue(`
  - line 1035: br := pmr.db.SendBatch(ctx, batch)
  - line 1039: _, err := br.Exec()
- **repo/postgres/postManagementMaster.go::CreatePostManagementMasterQuery** (lines 1676-1878)
  - line 1741: err := pmr.db.QueryRow(ctx, queryMaster,
  - line 1820: _, err = pmr.db.Exec(ctx, queryMaker,
- **repo/postgres/postManagementMaster.go::CreatePost** (lines 2103-2183)
  - line 2162: err := ap.db.QueryRow(dbCtx, insertPostQuery,
  - line 2174: _, err = ap.db.Exec(dbCtx, insertPostMappingQuery, postID)

## Missed by agent — REPO-002 (2)

| File | Function | Start | End | Hits |
|---|---|---|---|---|
| repo/postgres/postManagementMaster.go | CreatePostManagementMasterQuery | 1676 | 1878 | 2 |
| repo/postgres/posttopostmapping.go | ApprovePostMappingDetailMaker2 | 1322 | 1421 | 2 |

### Sample evidence

- **repo/postgres/postManagementMaster.go::CreatePostManagementMasterQuery** (lines 1676-1878)
  - line 1741: err := pmr.db.QueryRow(ctx, queryMaster,
  - line 1820: _, err = pmr.db.Exec(ctx, queryMaker,
- **repo/postgres/posttopostmapping.go::ApprovePostMappingDetailMaker2** (lines 1322-1421)
  - line 1362: cmdTag, err := tx.Exec(ctx, updateQuery, updateArgs...)
  - line 1408: row := tx.QueryRow(ctx, insertQuery, insertValues...)

## Missed by agent — HANDLER-001 (1)

| File | Function | Start | End | Hits |
|---|---|---|---|---|
| handler/reports.go | FetchAllPostsByDivisionHandler | 26 | 70 | 2 |

### Sample evidence

- **handler/reports.go::FetchAllPostsByDivisionHandler** (lines 26-70)
  - line 40: officeType, err := pmh.svc.GetOfficeTypeByID(ctx, req.OfficeID)
  - line 53: masterList, err := pmh.svc.FetchAllPostsByDivisionOfficeID(ctx, req.OfficeID)

## Extra findings (agent flagged, no deterministic signal)

_(none)_
