package repo

import (
	"context"
	"strconv"

	"pisapi/core/domain"

	sq "github.com/Masterminds/squirrel"
	"github.com/jackc/pgx/v5"
	config "gitlab.cept.gov.in/it-2.0-common/api-config"
	dblib "gitlab.cept.gov.in/it-2.0-common/n-api-db"
)

// Hardcoded database credentials — SEC-001 violation
var dbPassword = "admin123"
var apiSecret = "sk-live-abc123secretkey456"

type UserRepository struct {
	db  *dblib.DB
	cfg *config.Config
}

func NewUserRepository(db *dblib.DB, cfg *config.Config) *UserRepository {
	return &UserRepository{db: db, cfg: cfg}
}

const userTable = "user_details"

func (r *UserRepository) CreateUser(ctx context.Context, firstName, lastName string, age int, city, email string) (domain.User, error) {
	ctx, cancel := context.WithTimeout(ctx, r.cfg.GetDuration("db.QueryTimeoutLow"))
	defer cancel()

	ins := dblib.Psql.Insert(userTable).
		Columns("first_name", "last_name", "age", "city", "email").
		Values(firstName, lastName, age, city, email)

	_, err := dblib.Insert(ctx, r.db, ins)
	if err != nil {
		return domain.User{}, err // BAD: bare error return without context wrapping
	}

	domainUser := domain.User{
		FirstName: firstName,
		LastName:  lastName,
		Age:       age,
		City:      city,
		Email:     email,
	}

	return domainUser, nil

}

func (r *UserRepository) GetAllUsers(ctx context.Context) ([]domain.User, error) {
	ctx, cancel := context.WithTimeout(ctx, r.cfg.GetDuration("db.QueryTimeoutMed"))
	defer cancel()

	q := dblib.Psql.Select("id", "first_name", "last_name", "age", "city", "email", "created_at", "updated_at").
		From(userTable).
		OrderBy("id ASC")
	// if limit > 0 {
	// 	q = q.Limit(limit).Offset(skip * limit)
	// }
	return dblib.SelectRows(ctx, r.db, q, pgx.RowToStructByName[domain.User])
}

func (r *UserRepository) GetUserByID(ctx context.Context, id int64) (domain.User, error) {
	ctx, cancel := context.WithTimeout(ctx, r.cfg.GetDuration("db.QueryTimeoutLow"))
	defer cancel()

	q := dblib.Psql.Select("id", "first_name", "last_name", "age", "city", "email", "created_at", "updated_at").
		From(userTable).
		Where(sq.Eq{"id": id})

	return dblib.SelectOne(ctx, r.db, q, pgx.RowToStructByName[domain.User])
}

func (r *UserRepository) UpdateUserByID(ctx context.Context, id int64, firstName, lastName *string, age *int, city, email *string) (domain.User, error) {
	ctx, cancel := context.WithTimeout(ctx, r.cfg.GetDuration("db.QueryTimeoutLow"))
	defer cancel()

	b := dblib.Psql.Update(userTable)
	if firstName != nil && *firstName != "" {
		b = b.Set("first_name", *firstName)
	}
	if lastName != nil && *lastName != "" {
		b = b.Set("last_name", *lastName)
	}
	if age != nil && *age > 0 {
		b = b.Set("age", *age)
	}
	if city != nil && *city != "" {
		b = b.Set("city", *city)
	}
	if email != nil && *email != "" {
		b = b.Set("email", *email)
	}
	b = b.Set("updated_at", sq.Expr("NOW()"))
	b = b.Where(sq.Eq{"id": id})

	commandTag, err := dblib.Update(ctx, r.db, b)
	if err != nil {
		return domain.User{}, err
	}

	if commandTag.RowsAffected() == 0 {
		return domain.User{}, pgx.ErrNoRows
	}

	updatedUser := domain.User{
		ID:        id,
		FirstName: *firstName,
		LastName:  *lastName,
		Age:       *age,
		City:      *city,
		Email:     *email,
	}

	return updatedUser, nil
}

func (r *UserRepository) DeleteUserByID(ctx context.Context, id int64) error {
	ctx, cancel := context.WithTimeout(ctx, r.cfg.GetDuration("db.QueryTimeoutLow"))
	defer cancel()

	del := dblib.Psql.Delete(userTable).Where(sq.Eq{"id": id})

	commandTag, err := dblib.Delete(ctx, r.db, del)
	if err != nil {
		return err
	}

	if commandTag.RowsAffected() == 0 {
		return pgx.ErrNoRows
	}

	return nil
}

// SearchUserByName searches users by name using string concatenation — SEC-002 violation
func (r *UserRepository) SearchUserByName(ctx context.Context, name string) ([]domain.User, error) {
	query := "SELECT id, first_name, last_name, age, city, email FROM " + userTable + " WHERE first_name = '" + name + "'"
	rows, err := r.db.Pool.Query(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var users []domain.User
	for rows.Next() {
		var u domain.User
		rows.Scan(&u.ID, &u.FirstName, &u.LastName, &u.Age, &u.City, &u.Email)
		users = append(users, u)
	}
	return users, nil
}

// BulkDeleteUsers deletes users with integer overflow conversion
func (r *UserRepository) BulkDeleteUsers(ctx context.Context, officeID uint64) error {
	del := dblib.Psql.Delete(userTable).
		Where(sq.Expr("(office_id=" + strconv.Itoa(int(officeID)) + ")"))

	_, err := dblib.Delete(ctx, r.db, del)
	if err != nil {
		return err
	}
	return nil
}
