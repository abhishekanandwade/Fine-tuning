package handler

// import (
// 	"fmt"
// 	"regexp"

// 	//"io"
// 	"net/http"
// 	//"os"
// 	// "pmdm/config"
// 	_ "pmdm/docs"
// 	"sync/atomic"
// 	"time"

// 	"github.com/Depado/ginprom"
// 	"github.com/gin-contrib/cors"
// 	"github.com/gin-gonic/gin"
// 	swaggerFiles "github.com/swaggo/files"
// 	ginSwagger "github.com/swaggo/gin-swagger"
// 	authz "gitlab.cept.gov.in/it-2.0-common/api-authz"
// 	config "gitlab.cept.gov.in/it-2.0-common/api-config"
// 	apierrors "gitlab.cept.gov.in/it-2.0-common/api-errors"
// 	log "gitlab.cept.gov.in/it-2.0-common/api-log"
// )

// // Router is a wrapper for HTTP router
// type Router struct {
// 	*gin.Engine
// }

// var isShuttingDown atomic.Value

// func init() {
// 	isShuttingDown.Store(false)
// }

// // SetIsShuttingDown is an exported function that allows other packages to update the isShuttingDown value
// func SetIsShuttingDown(shuttingDown bool) {
// 	isShuttingDown.Store(shuttingDown)
// }

// func HealthCheckHandler(c *gin.Context) {
// 	shuttingDown := isShuttingDown.Load().(bool)
// 	if shuttingDown {
// 		// If the server is shutting down, respond with Service Unavailable
// 		c.JSON(http.StatusServiceUnavailable, gin.H{"status": "unhealthy"})
// 		return
// 	}
// 	// If the server is not shutting down, respond with OK
// 	c.JSON(http.StatusOK, gin.H{"status": "healthy"})
// }
// func ShutdownMiddleware() gin.HandlerFunc {
// 	shuttingDown := isShuttingDown.Load().(bool)
// 	return func(c *gin.Context) {
// 		if shuttingDown && c.Request.URL.Path != "/healthz" {
// 			c.AbortWithStatusJSON(http.StatusServiceUnavailable, gin.H{"error": "Server is shutting down"})
// 			return
// 		}
// 		c.Next()
// 	}
// }

// // NewRouter creates a new HTTP router for LMS
// func NewRouter(
// 	cfg *config.Config,
// 	//establishmentMasterHandler EstablishmentMasterHandler,
// 	cadreMasterHandler CadreMasterHandler,
// 	postManagementHandler PostManagementHandler,
// 	designationMasterHandler DesignationMasterHandler,
// 	posttoPostMappingrHandler PosttoPostMappingrHandler,

// ) (*Router, error) {
// 	// Disable debug mode and write logs to file in production
// 	//env := os.Getenv("APP_ENV")
// 	env := cfg.AppEnv()
// 	if env == "production" {
// 		gin.SetMode(gin.ReleaseMode)

// 		//logFile, _ := os.Create("gin.log")
// 		//gin.DefaultWriter = io.Writer(logFile)
// 	}

// 	// CORS configuration
// 	corsConfig := cors.DefaultConfig()
// 	corsConfig.AllowAllOrigins = true // Allow all origins for testing; replace with specific domains in production.
// 	corsConfig.AllowCredentials = true
// 	corsConfig.AllowHeaders = []string{"Content-Type", "Authorization", "Accept"}
// 	corsConfig.AllowMethods = []string{"GET", "POST", "PUT", "DELETE"} // Ensure OPTIONS is allowed for preflight

// 	router := gin.New()

// 	// Apply CORS middleware globally before routes
// 	router.Use(cors.New(corsConfig))
// 	router.Use(ShutdownMiddleware())

// 	// RedirectTrailingSlash to ensure URL consistency
// 	router.RedirectTrailingSlash = false

// 	// Custom error handling for undefined routes
// 	router.NoRoute(func(c *gin.Context) {
// 		c.JSON(http.StatusNotFound, gin.H{
// 			"success": false,
// 			"message": []string{"Invalid Path"},
// 			"errorno": []string{"INV1"},
// 		})
// 	})

// 	router.NoMethod(func(c *gin.Context) {
// 		c.JSON(http.StatusMethodNotAllowed, gin.H{
// 			"success": false,
// 			"message": []string{"Method Not Allowed"},
// 			"errorno": []string{"MD01"},
// 		})
// 	})

// 	router.Use(AuthMiddleware())

// 	// Apply environment-specific middleware
// 	if env == "production" {
// 		router.Use(gin.LoggerWithFormatter(customLogger), gin.Recovery(), cors.New(corsConfig), ValidateContentType([]string{"application/json"}))
// 	} else if env == "test" {
// 		router.Use(gin.LoggerWithFormatter(customLogger),
// 			gin.Recovery(),
// 			cors.New(corsConfig),
// 			ValidateContentType([]string{"application/json", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8", "text/css,*/*;q=0.1", "application/json,*/*", "*/*"}))
// 	}

// 	//router.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler)) // swagger call should be in v1
// 	router.GET("/healthz", HealthCheckHandler)
// 	/*metrics*/
// 	p := ginprom.New(
// 		ginprom.Engine(router),
// 		ginprom.Subsystem("gin"),
// 		ginprom.Path("/metrics"),
// 	)
// 	router.Use(p.Instrument())
// 	/*metrics*/

// 	v1 := router.Group("v1")
// 	{
// 		v1.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerFiles.Handler))

// 		pm := v1.Group("/post-management")
// 		{
// 			// Group for cadre operations based on cadre-id
// 			cadre := pm.Group("/cadres") //3
// 			{
// 				cadre.GET("", cadreMasterHandler.ListCadresHandler)                                                //query parameter as group-code, make response struct uniform                                // List all cadres
// 				cadre.GET("/:cadre-id", postManagementHandler.GetPostNameMasterHandler)                            // Fetch post names by cadre ID
// 				cadre.GET("/cadre-groups", postManagementHandler.PostManagementGroupByCadreCountByOfficeIDHandler) // Group post by cadre count and office ID
// 				//TBD no other life cycle other APIs are supported.
// 			}

// 			// Group for managing posts by cadre and office
// 			posts := pm.Group("/posts") //8
// 			{
// 				//posts.POST("", postManagementHandler.CreatePostManagementMaster)               // Create a new post management master
// 				posts.GET("", postManagementHandler.PostManagementByOfficeAndPostHandler)             //m1             //cadre-name in query // Fetch post by cadre name and office ID
// 				posts.GET("/:post-id", postManagementHandler.GetPostManagementMasterWithMakerHandler) // Get post management master by post ID
// 				posts.GET("/:post-id/surplus", postManagementHandler.FetchSurplusPostRecordByApproverPostIDHandler)
// 				//posts.PUT("/:post-id", postManagementHandler.UpdatePostManagementMaster)                            // Update a post management master by post ID
// 				//posts.PUT("/:post-id/approve", postManagementHandler.ApprovePostManagementMaster)                   // Approve post management master
// 				posts.PUT("/:post-id/status-change", postManagementHandler.PostManagementChangFilledStatusByPostIDHandler) // Change filled status of a post
// 				//posts.DELETE("/:post-id", postManagementHandler.DeletePostManagementMaster)                         // Delete a post management master by post ID                   // Restore a surplus post
// 				posts.POST("/surplus-restore-bulk", postManagementHandler.RestoredSurplusPostHandler) //updated-by from body
// 				//TBD:by all query we may need to support all the posts created in a given period/unit...
// 			}

// 			pstmOffice := pm.Group("/office-post-details") //2 it cant be a resource
// 			{
// 				pstmOffice.GET("/:office-id/posts", postManagementHandler.PostManagementByOfficeIDAndStatusHandler) //keep only super set// Fetch post by office ID, status, and group ID
// 				pstmOffice.GET("/:office-id/posts-summary", postManagementHandler.FetchPostsByOfficeIDHandler)      // merged below two move to posts
// 				//pstmOffice.GET("/:office-id/cadre-posts",postManagementHandler.PostManagementByCadreAndOffice)
// 			}

// 			pstmMaker := pm.Group("/makers") //7
// 			{
// 				pstmMaker.POST("", postManagementHandler.CreatePostManagementMakerHandler)                            // Create a new post management maker
// 				pstmMaker.GET("/status-pending", postManagementHandler.PostManagementWithPendingStatusOfMakerHandler) //
// 				pstmMaker.GET("", postManagementHandler.FetchPostsByOfficeIDAndMakerHandler)
// 				pstmMaker.POST("/approve-bulk", postManagementHandler.ApprovePostManagementMakerHandler)                      // Approve a post management maker
// 				pstmMaker.POST("/abolish-bulk", postManagementHandler.ApprovePostManagementMakerForAbolishPostHandler)        // Approve abolition of a post by maker ID
// 				pstmMaker.POST("/exchange-post-bulk", postManagementHandler.ApprovePostManagementMakerForExchangePostHandler) // Approve exchange of a post by maker ID
// 				pstmMaker.POST("/reject-bulk", postManagementHandler.RejectPostManagementMakerHandler)
// 			}

// 			// pstmFile := pm.Group("/files") //2
// 			// {
// 			// 	pstmFile.POST("", postManagementHandler.UploadFile)   // Upload a file
// 			// 	pstmFile.GET("", postManagementHandler.DownloadFiles) // Download files //file-path
// 			// }

// 			pstmEstablishment := pm.Group("/office-establishment-register") //3
// 			{
// 				pstmEstablishment.POST("", postManagementHandler.CreateEstablishmentRegisterHandler) // Create a new establishment register
// 				pstmEstablishment.GET("", postManagementHandler.FetchEstablishmentRegisterHandler)   // merged below two
// 				pstmEstablishment.GET("/:office-id/establishments", postManagementHandler.EstblishnentRegisterByOfficeHandler)
// 			}

// 			dsm := pm.Group("/designations") // 1 these two be merged with super set response
// 			{
// 				dsm.GET("", designationMasterHandler.ListAndFilterDesignationsHandler) //query parameters // Fetch cadres by group and cadre code
// 				//dsm.GET("", designationMasterHandler.ListDesignations) delete this an dmak esuperset for above                                                      // List all designations
// 			}

// 			ptopmap := pm.Group("/ptop-mappings") //7
// 			{

// 				ptopmap.GET("", posttoPostMappingrHandler.GetPostMappingMasterHandler)                                          // List all mappings
// 				ptopmap.GET("/:post-id/authority-details", posttoPostMappingrHandler.GetAuthorityDetailsByPostIDHandler)        //query param
// 				ptopmap.GET("/:post-id/master-authority-details", posttoPostMappingrHandler.GetMasterAuthoritiesDeatilsHandler) //get query param
// 				//ptopmap.PUT("/:post-id", posttoPostMappingrHandler.UpdatePostMappingDetail)                              // Update mapping by employee post ID
// 				ptopmap.GET("/posts/:post-id/roles/:role-id", posttoPostMappingrHandler.GetMulAuthIDHandler)    //whether to be used              // Fetch post-role mapping
// 				ptopmap.POST("/posts-bulk", posttoPostMappingrHandler.UpdateArrayOfEmpPostIDForParticularField) // Bulk update employees for posts

// 			}
// 			postMappingMaker := pm.Group("/ptop-mappings-makers") //3
// 			{
// 				postMappingMaker.POST("", posttoPostMappingrHandler.CreatePostMappingDetailMaker)              // Create a new post mapping maker
// 				postMappingMaker.GET("", posttoPostMappingrHandler.GetPostMappingMasterMaker)                  //post-id query param
// 				postMappingMaker.POST("approve-bulk", posttoPostMappingrHandler.ApprovePostMappingDetailMaker) // Approve post mapping by maker ID
// 			}
// 		}

// 	}

// 	return &Router{
// 		router,
// 	}, nil
// }

// // }

// // Serve starts the HTTP server
// func (r *Router) Serve(listenAddr string) error {
// 	return r.Run(listenAddr)
// }

// // customLogger is a custom Gin logger
// func customLogger(param gin.LogFormatterParams) string {
// 	return fmt.Sprintf("[%s] - %s \"%s %s %s %d %s [%s]\"\n",
// 		param.TimeStamp.Format(time.RFC1123),
// 		param.ClientIP,
// 		param.Method,
// 		param.Path,
// 		param.Request.Proto,
// 		param.StatusCode,
// 		param.Latency.Round(time.Millisecond),
// 		param.Request.UserAgent(),
// 	)
// }

// func ValidateContentType(allowedTypes []string) gin.HandlerFunc {
// 	return func(c *gin.Context) {

// 		contentType := c.GetHeader("Content-Type")
// 		//var contentType string
// 		//contentType = c.GetHeader("Accept")
// 		// Check if the Content-Type is in the allowedTypes
// 		validContentType := false
// 		for _, allowedType := range allowedTypes {
// 			if contentType == allowedType {
// 				validContentType = true
// 				break
// 			}
// 		}

// 		if !validContentType {
// 			c.JSON(http.StatusUnsupportedMediaType, gin.H{
// 				"success": false,
// 				"message": []string{"Invalid Content-Type. Supported types are: " + fmt.Sprintf("%v", allowedTypes)},
// 				"errorno": []string{"USP1"},
// 			})
// 			c.Abort()
// 			return
// 		}

// 		c.Next()
// 	}

// }

// // List of paths to be ignored for authorization (regular expressions)
// var ignoredPaths = []string{
// 	".*/healthz$",
// 	".*/swagger/.*$",
// 	".*/metrics/.*$",
// }

// func AuthMiddleware() gin.HandlerFunc {
// 	return func(c *gin.Context) {
// 		// Check if the current request path matches any of the ignored paths
// 		for _, pathPattern := range ignoredPaths {
// 			matched, err := regexp.MatchString(pathPattern, c.Request.URL.Path)
// 			if err != nil {
// 				log.Error(c, "Error matching ignored path pattern: %v", err)
// 				apierrors.HandleValidationError(c, err)
// 				c.Abort()
// 				return
// 			}
// 			if matched {
// 				log.Info(c, "Path %s is ignored for authorization", c.Request.URL.Path)
// 				c.Next() // Skip authorization and proceed to the next middleware or handler
// 				return
// 			}
// 		}

// 		// Proceed with authorization if the path is not ignored
// 		authResult, authErr := authz.Authorize(c)
// 		if authErr != nil {
// 			log.Error(c, "Authorization error: %v", authErr)

// 			// Check if authErr is of type AppError to access the Code field
// 			if appErr, ok := authErr.(*apierrors.AppError); ok {
// 				if appErr.Code == "401" {
// 					apierrors.HandleUnauthorizedError(c)
// 				} else if appErr.Code == "500" {
// 					apierrors.HandleError(c, appErr)
// 				}
// 			} else {
// 				apierrors.HandleError(c, authErr) // Handle non-AppError errors generically
// 			}

// 			c.Abort()
// 			return
// 		}
// 		if !authResult.Authorization {
// 			log.Info(c, "Access denied: User %s is not authorized", c.GetHeader("X-User-ID"))
// 			apierrors.HandleForbiddenError(c)
// 			c.Abort()
// 			return
// 		}

// 		c.Next() // Proceed to the next middleware or request handler
// 	}
// }
