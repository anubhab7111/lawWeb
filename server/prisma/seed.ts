import 'dotenv/config';
import { PrismaClient } from '@prisma/client';
import { PrismaPg } from '@prisma/adapter-pg';

const adapter = new PrismaPg({ connectionString: process.env.DATABASE_URL });
const prisma = new PrismaClient({ adapter });

const lawyers = [
    {
        id: '1',
        name: 'Sarah Mitchell',
        specialty: 'Criminal Defense',
        experience: 15,
        rating: 4.9,
        hourlyRate: 350,
        location: 'New York, NY',
        bio: 'Experienced criminal defense attorney with a proven track record in white-collar crimes, DUI cases, and federal offenses.',
        cases: 450,
        successRate: 92,
        education: 'Harvard Law School, JD',
        languages: ['English', 'Spanish'],
        availability: 'Available this week',
    },
    {
        id: '2',
        name: 'David Chen',
        specialty: 'Family Law',
        experience: 12,
        rating: 4.8,
        hourlyRate: 275,
        location: 'Los Angeles, CA',
        bio: 'Compassionate family lawyer specializing in divorce, child custody, and adoption cases with a focus on mediation.',
        cases: 380,
        successRate: 88,
        education: 'Stanford Law School, JD',
        languages: ['English', 'Mandarin'],
        availability: 'Available next week',
    },
    {
        id: '3',
        name: 'Jennifer Rodriguez',
        specialty: 'Business & Corporate Law',
        experience: 18,
        rating: 4.9,
        hourlyRate: 425,
        location: 'Chicago, IL',
        bio: 'Corporate attorney specializing in mergers, acquisitions, and business formation with Fortune 500 experience.',
        cases: 520,
        successRate: 95,
        education: 'Yale Law School, JD',
        languages: ['English', 'Spanish'],
        availability: 'Available in 2 weeks',
    },
    {
        id: '4',
        name: 'Michael Thompson',
        specialty: 'Personal Injury',
        experience: 10,
        rating: 4.7,
        hourlyRate: 300,
        location: 'Houston, TX',
        bio: 'Dedicated personal injury lawyer fighting for accident victims and securing maximum compensation.',
        cases: 340,
        successRate: 90,
        education: 'University of Texas Law School, JD',
        languages: ['English'],
        availability: 'Available this week',
    },
    {
        id: '5',
        name: 'Emily Watson',
        specialty: 'Real Estate Law',
        experience: 14,
        rating: 4.8,
        hourlyRate: 325,
        location: 'Miami, FL',
        bio: 'Real estate attorney handling residential and commercial transactions, zoning issues, and property disputes.',
        cases: 410,
        successRate: 93,
        education: 'Columbia Law School, JD',
        languages: ['English', 'Portuguese'],
        availability: 'Available next week',
    },
];

async function main() {
    for (const lawyer of lawyers) {
        await prisma.lawyer.upsert({
            where: { id: lawyer.id },
            update: lawyer,
            create: lawyer,
        });
    }
    console.log(`Seeded ${lawyers.length} lawyers`);
}

main()
    .catch((e) => {
        console.error(e);
        process.exit(1);
    })
    .finally(() => prisma.$disconnect());
