package router

import (
	//"github.com/gin-gonic/gin"
	"net/http"
	handler "pmdm/handler"
	"sync/atomic"

	"github.com/gin-gonic/gin"
	swaggerFiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
	apierrors "gitlab.cept.gov.in/it-2.0-common/api-errors"
	r "gitlab.cept.gov.in/it-2.0-common/api-server"
)

var isShuttingDown atomic.Value

func init() {
	isShuttingDown.Store(false)
}

// SetIsShuttingDown is an exported function that allows other packages to update the isShuttingDown value
// func SetIsShuttingDown(shuttingDown bool) {
// 	isShuttingDown.Store(shuttingDown)
// }

func HealthCheckHandler(c *gin.Context) {
	shuttingDown := isShuttingDown.Load().(bool)
	if shuttingDown {
		// If the server is shutting down, respond with Service Unavailable
		c.JSON(http.StatusServiceUnavailable, gin.H{"status": "unhealthy"})
		return
	}
	// If the server is not shutting down, respond with OK
	c.JSON(http.StatusOK, gin.H{"status": "healthy"})
}

func Routes(router *r.Router,
	cadreMasterHandler *handler.CadreMasterHandler,
	postManagementHandler *handler.PostManagementHandler,
	designationMasterHandler *handler.DesignationMasterHandler,
	posttoPostMappingrHandler *handler.PosttoPostMappingrHandler,
) {

	router.NoRoute(func(ctx *gin.Context) {
		apierrors.HandleNoRouteError(ctx)
	})
	router.NoMethod(func(ctx *gin.Context) {
		apierrors.HandleNoMethodError(ctx)
	})
	router.GET("/healthz", HealthCheckHandler)
	v1 := router.Group("v1")
	{
		v1.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))
		pm := v1.Group("/post-management")
		{
			cadre := pm.Group("/cadres")
			{
				cadre.POST("/create", cadreMasterHandler.CreateCadreMasterHandler)
				cadre.GET("", cadreMasterHandler.ListCadresHandler)
				cadre.PUT("/:cadre-id", cadreMasterHandler.UpdateCadreMasterHandler)
				// cadre.GET("/:cadre-id", postManagementHandler.GetPostNameMasterHandler)
				cadre.GET("/cadre-groups", postManagementHandler.PostManagementGroupByCadreCountByOfficeIDHandler)
				//TBD no other life cycle other APIs are supported.
				cadre.GET("/all-cadres", cadreMasterHandler.ListAllCadresHandler)
				cadre.GET("/all-cadres-d1", cadreMasterHandler.ListAllCadresHandlerD1)
			}

			posts := pm.Group("/posts") //8
			{
				posts.POST("/create", postManagementHandler.CreatePostManagementMasterHandler)
				posts.GET("", postManagementHandler.PostManagementByOfficeAndPostHandler)
				posts.GET("/d1", postManagementHandler.PostManagementByOfficeAndPostHandlerD1)
				posts.GET("/:post-id", postManagementHandler.GetPostManagementMasterWithMakerHandler)
				posts.GET("/:post-id/surplus", postManagementHandler.FetchSurplusPostRecordByApproverPostIDHandler)
				//posts.PUT("/:post-id", postManagementHandler.UpdatePostManagementMaster)                            // Update a post management master by post ID
				posts.PUT("/:post-id/status-change", postManagementHandler.PostManagementChangFilledStatusByPostIDHandler) // Change filled status of a post
				//posts.DELETE("/:post-id", postManagementHandler.DeletePostManagementMaster)                         // Delete a post management master by post ID                   // Restore a surplus post
				posts.POST("/surplus-restore-bulk", postManagementHandler.RestoredSurplusPostHandler) //updated-by from body
				posts.POST("/approve", postManagementHandler.ApprovePostManagementMasterWithMaker)    // Approve post management master

				posts.PUT("/filled-status", postManagementHandler.PostManagementChangFilledStatusHandler)
				posts.GET("/division-office", postManagementHandler.ListPostManagementHandler)

				posts.GET("/available-posts", postManagementHandler.ListAvailablePostsHandler)
				posts.GET("/vacant-posts", postManagementHandler.ListVacantPostsHandler)

				posts.GET("/group-master", postManagementHandler.ListGroupMasterHandler)
				posts.GET("/office-details", postManagementHandler.ListOfficeDetailsHandler)
				posts.GET("/group-cadre", postManagementHandler.ListGroupCadreHandler)
				posts.POST("/bulk/create", postManagementHandler.CreatePostHandler)

				posts.PUT("/update-pmm", postManagementHandler.UpdatePostHandler) // Update a post management master by post ID

				posts.GET("/post-details", postManagementHandler.GetPostDetailsHandler)
				posts.GET("/cl-granting-posts", postManagementHandler.GetCLGrantingPostsHandler)
				posts.GET("/ddo-filtered-posts", postManagementHandler.GetDDOFilteredPostsHandler)
				posts.GET("/post-details/:post-id", postManagementHandler.GetPostDetailsbyPostIDHandler)
				posts.PUT("/post-details-update", postManagementHandler.UpdatePostDetailsbyPostIDHandler)
				posts.GET("/check-posts-status", postManagementHandler.CheckPostsStatusHandler)
				posts.DELETE("/delete-by-office-id", postManagementHandler.DeletePostsbyOfficeIDHandler)
				posts.GET("/check-cadres", postManagementHandler.CheckCadreExistsHandler)
				posts.GET("/sanctioned-strength", postManagementHandler.GetSanctionedStrengthByOfficeIDHandler)
				posts.PUT("/post-name-update", postManagementHandler.UpdatePostnameByPostIdByHandler)

				posts.GET("/get-post/:post-id", postManagementHandler.GetPostDetailsByPostIdHandler)

				posts.GET("summary", postManagementHandler.GetPostManagementSummaryHandler)
				posts.GET("/circles", postManagementHandler.GetPostManagemmentSummaryHandler)
				posts.GET("/regions", postManagementHandler.GetCircleSummaryHandler)
				posts.GET("/divisions", postManagementHandler.GetRegionSummaryHandler)
				posts.GET("/offices", postManagementHandler.GetDivisionSummaryHandler)
			}

			pstmOffice := pm.Group("/office-post-details") //2 it cant be a resource
			{
				pstmOffice.GET("/:office-id/posts", postManagementHandler.PostManagementByOfficeIDAndStatusHandler) //keep only super set// Fetch post by office ID, status, and group ID
				pstmOffice.GET("/:office-id/posts-summary", postManagementHandler.FetchPostsByOfficeIDHandler2)     // merged below two move to posts
				//pstmOffice.GET("/:office-id/cadre-posts",postManagementHandler.PostManagementByCadreAndOffice)
				pstmOffice.GET("", postManagementHandler.PostManagementByOfficeIDHandler)                              // Fetch post by office ID and status
				pstmOffice.GET("/:office-id/posts-summary-all", postManagementHandler.FetchAllPostsByOfficeIDHandler2) // merged below two move to posts
				pstmOffice.GET("/:office-id/post-details", postManagementHandler.GetPostDetailHandler)
				pstmOffice.GET("/:office-id/division-report", postManagementHandler.FetchAllPostsByDivisionHandler)
			}

			pstmMaker := pm.Group("/makers") //7
			{
				pstmMaker.POST("", postManagementHandler.CreatePostManagementMakerHandler)                            // Create a new post management maker
				pstmMaker.GET("/status-pending", postManagementHandler.PostManagementWithPendingStatusOfMakerHandler) //
				pstmMaker.GET("", postManagementHandler.FetchPostsByOfficeIDAndMakerHandler)
				pstmMaker.POST("/approve-bulk", postManagementHandler.ApprovePostManagementMakerHandler)                      // Approve a post management maker
				pstmMaker.POST("/abolish-bulk", postManagementHandler.ApprovePostManagementMakerForAbolishPostHandler)        // Approve abolition of a post by maker ID
				pstmMaker.POST("/exchange-post-bulk", postManagementHandler.ApprovePostManagementMakerForExchangePostHandler) // Approve exchange of a post by maker ID
				pstmMaker.POST("/reject-bulk", postManagementHandler.RejectPostManagementMakerHandler)
				pstmMaker.GET("/approve-create-post", postManagementHandler.FetchPendingCreatePostApprovalsByOfficeIDHandler) // Fetch pending create post approvals by office ID
			}

			pstmFile := pm.Group("/files") //2
			{
				pstmFile.POST("/upload", postManagementHandler.UploadFile) // Upload a file
				pstmFile.GET("", postManagementHandler.DownloadFiles)      // Download files //file-path
			}

			pstmEstablishment := pm.Group("/office-establishment-register") //3
			{
				pstmEstablishment.POST("", postManagementHandler.CreateEstablishmentRegisterHandler) // Create a new establishment register
				pstmEstablishment.GET("", postManagementHandler.FetchEstablishmentRegisterHandler)   // merged below two
				pstmEstablishment.GET("/:office-id/establishments", postManagementHandler.EstblishnentRegisterByOfficeHandler)
			}

			dsm := pm.Group("/designations") // 1 these two be merged with super set response
			{
				dsm.GET("", designationMasterHandler.ListAndFilterDesignationsHandler)                     //query parameters // Fetch cadres by group and cadre code
				dsm.POST("/create", designationMasterHandler.CreateDesignationMasterHandler)               // Create a new designation
				dsm.PUT("/:designation-uid", designationMasterHandler.UpdateDesignationMasterHandler)      //query parameters // Fetch cadres by group and cadre code
				dsm.GET("/cadres-by-group", designationMasterHandler.ListCadresByGroupHandler)             //query parameters // Fetch cadres by group and cadre code
				dsm.GET("/designations-by-cadre", designationMasterHandler.ListDesignationsByCadreHandler) //query parameters // Fetch cadres by group and cadre code
				dsm.GET("/all-designations", designationMasterHandler.ListAllDesignationsHandler)
				dsm.GET("/d1", designationMasterHandler.ListDesignationD1)
			}

			ptopmap := pm.Group("/ptop-mappings") //7
			{
				ptopmap.GET("", posttoPostMappingrHandler.GetPostMappingMasterHandler)                                          // List all mappings
				ptopmap.GET("/:post-id/authority-details", posttoPostMappingrHandler.GetAuthorityDetailsByPostIDHandler)        //query param
				ptopmap.GET("/:post-id/master-authority-details", posttoPostMappingrHandler.GetMasterAuthoritiesDeatilsHandler) //get query param
				ptopmap.GET("/posts/:post-id/roles/:role-id", posttoPostMappingrHandler.GetMulAuthIDHandler)                    //whether to be used              // Fetch post-role mapping
				ptopmap.POST("/posts-bulk", posttoPostMappingrHandler.UpdateArrayOfEmpPostIDForParticularField)
				ptopmap.POST("/hoo", posttoPostMappingrHandler.IdentifyHeadOfOffice)              // Bulk update employees for posts
				ptopmap.POST("/head-office/update", posttoPostMappingrHandler.UpdateHeadOfOffice) // Bulk update employees for posts
				ptopmap.GET("/head-office/search", posttoPostMappingrHandler.GetHeadPostOccupant) // Bulk update employees for posts
				ptopmap.GET("/post/search", posttoPostMappingrHandler.GetPostDetailsForHOO)       // Bulk update employees for posts

				ptopmap.GET("/post-redeployment/:office-id/:cadre-name", posttoPostMappingrHandler.GetPostRedeplomentByOfficeIDHandler) //Post Redepolyment
				ptopmap.POST("/save-post-redployment", posttoPostMappingrHandler.SavePostRedeploymentHandler)                           //Post Redepolyment
				ptopmap.GET("/circlenames", posttoPostMappingrHandler.GetCircleOfficeIDsHandler)                                        //Post Redepolyment
				ptopmap.GET("/regionnames/:circle-id", posttoPostMappingrHandler.GetRegionalOfficeIDsHandler)                           //Post Redepolyment
				ptopmap.GET("/divisionnames/:region-id", posttoPostMappingrHandler.GetDivisionalOfficeIDsHandler)                       //Post Redepolyment
				ptopmap.GET("/cadrenames", posttoPostMappingrHandler.GetCadreDetailsHandler)
				ptopmap.GET("/postnamees", posttoPostMappingrHandler.GetPostDetailsHandler)
				ptopmap.GET("/designationnames", posttoPostMappingrHandler.GetDesignationDetailsHandler)
				ptopmap.GET("/post-details/:post-id", posttoPostMappingrHandler.GetPostDetailsForRedeploymentHandler) //Post Redepolyment
				ptopmap.GET("/post-redeployed-inward-reports/:office-id", posttoPostMappingrHandler.GetPostRedeployedInwardReportsHandler)
				ptopmap.GET("/post-redeployed-outward-reports/:office-id", posttoPostMappingrHandler.GetPostRedeployedOutwardReportsHandler)
				ptopmap.POST("/save-post-redeployment2", posttoPostMappingrHandler.SavePostRedeploymentHandler2)
				ptopmap.GET("/redeployed-post-authority-details/:post-id", posttoPostMappingrHandler.GetRedeployedPostAuthorityChargesHandler)
				ptopmap.PUT("/update-redeployed-post-authority-details/:post-id", posttoPostMappingrHandler.UpdateRedeployedPostAuthorityChargesHandler)
			}
			postMappingMaker := pm.Group("/ptop-mappings-makers") //3
			{
				postMappingMaker.POST("", posttoPostMappingrHandler.CreatePostMappingDetailMaker) // Create a new post mapping maker
				postMappingMaker.GET("", posttoPostMappingrHandler.GetPostMappingMasterMaker)     //post-id query param
				postMappingMaker.GET("/:approve-post-id", posttoPostMappingrHandler.GetPostMappingMakerDetails)
				postMappingMaker.POST("approve-bulk", posttoPostMappingrHandler.ApprovePostMappingDetailMaker) // Approve post mapping by maker ID
				// postMappingMaker.POST("/hierarchy-posts", posttoPostMappingrHandler.GetPostAndEmployeeHierarchy)
				postMappingMaker.POST("/save", posttoPostMappingrHandler.SavePostToPostMappings)
				postMappingMaker.GET("/hierarchy-posts/:office-id", posttoPostMappingrHandler.GetPostAndEmployeeHierarchyHandler)
			}
			reports := pm.Group("/reports")
			{
				reports.GET("/posts-filled-vacant-status/:office-id", postManagementHandler.GetPostsFilledVacantStatusHandler)
				reports.GET("/posts-created-redeployed-abolished/:year/:month", postManagementHandler.GetPostsCreatedRedeployedAbolishedHandler)
				reports.GET("/posts-filled-vacant-status-detailed/:office-id", postManagementHandler.GetPostsFilledVacantStatusDetailedHandler)
				reports.GET("/exception-report-order-casemark", postManagementHandler.ExceptionReportOrderCasemarkHandler)
				// reports.GET("/exception-report-establishment-register", postManagementHandler.ExceptionReportEstablishmentRegisterHandler)
				// reports.GET("/exception-report-office-name", postManagementHandler.ExceptionReportOfficeNameHandler)
				// reports.GET("/exception-report-cadre-name", postManagementHandler.ExceptionReportCadreNameHandler)
				// reports.GET("/exception-report-group-name", postManagementHandler.ExceptionReportGroupNameHandler)
				reports.GET("/cadre-wise-reports/:office-id", postManagementHandler.GetCadreWiseReportsHandler)
				reports.GET("/cadre-wise-reports/:office-id/:cadre-id", postManagementHandler.GetCadreWiseOfficeWiseReportsHandler)
				reports.GET("/cadres", postManagementHandler.GetCadreReport)
				reports.GET("/circle", postManagementHandler.GetCircleCadreReport)
				reports.GET("/divisions", postManagementHandler.GetDivisionsReport)
				reports.GET("/hierarchy", postManagementHandler.GetHierarchyReport)
				reports.GET("/offices", postManagementHandler.GetOfficeReport)
				reports.GET("posts", postManagementHandler.GetPostReport)
				reports.GET("/regions", postManagementHandler.GetRegionHandler)
				reports.GET("/list-cadre-wise-reports/:division-id/:cadre-id", postManagementHandler.GetListCadreWiseOfficeWiseReportsHandler)
				reports.GET("/post-authority-details/:office-id", postManagementHandler.GetPostAuthorityChargesDetailsHandler)
			}
		}
	}
}
