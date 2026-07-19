import express from 'express';
import { prisma } from '../db';

const router = express.Router();

// Get all lawyers
router.get('/', async (req, res) => {
    try {
        const lawyers = await prisma.lawyer.findMany({ orderBy: { id: 'asc' } });
        res.json(lawyers);
    } catch (err: any) {
        res.status(500).json({ message: err.message });
    }
});

// Get lawyer by ID
router.get('/:id', async (req, res) => {
    try {
        const lawyer = await prisma.lawyer.findUnique({ where: { id: req.params.id } });
        if (!lawyer) return res.status(404).json({ message: 'Lawyer not found' });
        res.json(lawyer);
    } catch (err: any) {
        res.status(500).json({ message: err.message });
    }
});

// Recommend lawyers
router.post('/recommend', async (req, res) => {
    // Basic recommendation logic: match specialty
    const { problemDescription, specialty } = req.body;

    // This is where we could integrate AI to better match descriptions to specialties
    // For now, simple filter

    try {
        let filteredLawyers = await prisma.lawyer.findMany({ orderBy: { id: 'asc' } });
        if (specialty) {
            filteredLawyers = filteredLawyers.filter(lawyer =>
                lawyer.specialty.toLowerCase().includes(specialty.toLowerCase())
            );
        }

        res.json(filteredLawyers);
    } catch (err: any) {
        res.status(500).json({ message: err.message });
    }
});

export default router;
