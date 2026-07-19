import 'dotenv/config';
import { defineConfig } from 'prisma/config';

// Prisma 7: connection URLs live here (for the CLI/Migrate), not in
// schema.prisma. The runtime client gets its connection via the pg
// driver adapter in src/db.ts.
export default defineConfig({
    schema: 'prisma/schema.prisma',
    migrations: {
        path: 'prisma/migrations',
        seed: 'ts-node prisma/seed.ts',
    },
    datasource: {
        url: process.env.DATABASE_URL!,
    },
});
