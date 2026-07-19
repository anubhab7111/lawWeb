// Load .env before reading DATABASE_URL: this module is imported (and
// evaluated) before index.ts gets a chance to call dotenv.config().
import 'dotenv/config';
import { PrismaClient } from '@prisma/client';
import { PrismaPg } from '@prisma/adapter-pg';

// Prisma 7: the client connects through a driver adapter; the connection
// string comes from .env (DATABASE_URL -> local PostgreSQL).
const adapter = new PrismaPg({ connectionString: process.env.DATABASE_URL });

// Single shared Prisma client for the whole server.
export const prisma = new PrismaClient({ adapter });
