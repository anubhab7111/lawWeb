import mongoose, { Schema, Document } from 'mongoose';

export interface ILawyer extends Document {
    name: string;
    specialty: string;
    experience: number;
    rating: number;
    hourlyRate: number;
    location: string;
    bio: string;
    cases: number;
    successRate: number;
    education: string;
    languages: string[];
    availability: string;
}

const LawyerSchema: Schema = new Schema({
    name: { type: String, required: true },
    specialty: { type: String, required: true },
    experience: { type: Number, required: true },
    rating: { type: Number, required: true },
    hourlyRate: { type: Number, required: true },
    location: { type: String, required: true },
    bio: { type: String, required: true },
    cases: { type: Number, required: true },
    successRate: { type: Number, required: true },
    education: { type: String, required: true },
    languages: [{ type: String }],
    availability: { type: String, required: true }
});

export default mongoose.model<ILawyer>('Lawyer', LawyerSchema);
