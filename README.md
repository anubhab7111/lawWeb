# Law Education Website with Chatbot

## Local PostgreSQL setup (required by the Express server)

The Express API (`server/src`) persists users, lawyers, and bookings to a locally-run PostgreSQL via Prisma.

One-time setup (Arch Linux):

```bash
sudo pacman -S --needed postgresql
sudo -u postgres initdb --locale=en_US.UTF-8 -E UTF8 -D /var/lib/postgres/data   # skip if already initialized
sudo systemctl enable --now postgresql
sudo -u postgres psql -c "CREATE ROLE lawweb LOGIN PASSWORD 'lawweb' CREATEDB;"
sudo -u postgres createdb -O lawweb lawweb
```

Add to `server/.env`:

```
DATABASE_URL="postgresql://lawweb:lawweb@localhost:5432/lawweb?schema=public"
```

Then create the tables and seed the lawyer directory:

```bash
cd server
npm install
npx prisma migrate dev   # applies prisma/migrations
npx prisma db seed       # seeds the 5-lawyer directory
npm run dev              # Express on :5001 (expects "Connected to PostgreSQL")
```

The Python FastAPI service (`server/app`, port 8000) does not use the database; run it with `python run.py` as before.
