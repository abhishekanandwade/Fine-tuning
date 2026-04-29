package port

var (
	ListSuccess     StatusCodeAndMessage = StatusCodeAndMessage{StatusCode: 200, Message: "list retrieved successfully", Success: true}
	FetchSuccess    StatusCodeAndMessage = StatusCodeAndMessage{StatusCode: 200, Message: "data retrieved successfully", Success: true}
	CreateSuccess   StatusCodeAndMessage = StatusCodeAndMessage{StatusCode: 201, Message: "resource created successfully", Success: true}
	UpdateSuccess   StatusCodeAndMessage = StatusCodeAndMessage{StatusCode: 200, Message: "resource updated successfully", Success: true}
	DeleteSuccess   StatusCodeAndMessage = StatusCodeAndMessage{StatusCode: 200, Message: "resource deleted successfully", Success: true}
	TransferSuccess StatusCodeAndMessage = StatusCodeAndMessage{StatusCode: 200, Message: "post redeployed successfully", Success: true}
)

type StatusCodeAndMessage struct {
	StatusCode int    `json:"status_code"`
	Success    bool   `json:"success"`
	Message    string `json:"message"`
}

type MetaDataResponse struct {
	Skip                 uint64 `json:"skip,default=0"`
	Limit                uint64 `json:"limit,default=10"`
	OrderBy              string `json:"order_by,omitempty"`
	SortType             string `json:"sort_type,omitempty"`
	TotalRecordsCount    int    `json:"total_records_count,omitempty"`
	ReturnedRecordsCount int    `json:"returned_records_count"`
}

func NewMetaDataResponse(skip, limit uint64, total int) MetaDataResponse {
	return MetaDataResponse{
		Skip:                 skip,
		Limit:                limit,
		ReturnedRecordsCount: total,
	}
}

type APIResponse struct {
	StatusCodeAndMessage
	Data interface{} `json:"data,omitempty"`
}
