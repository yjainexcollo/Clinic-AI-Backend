// MongoDB initialization script for clinicai database
// This script runs automatically when MongoDB container starts for the first time

print('==================================================');
print('🚀 Starting MongoDB initialization for clinicai database');
print('==================================================');

// Get credentials from environment variables set by Docker
const adminUser = process.env.MONGO_INITDB_ROOT_USERNAME || 'admin';
const adminPassword = process.env.MONGO_INITDB_ROOT_PASSWORD || 'password';
const dbName = process.env.MONGO_INITDB_DATABASE || 'clinicai';

print(`Creating root user: ${adminUser}`);

// Switch to admin database to create root user
db = db.getSiblingDB('admin');

// Create the root admin user with full privileges
try {
    db.createUser({
        user: adminUser,
        pwd: adminPassword,
        roles: [
            { role: 'root', db: 'admin' }
        ]
    });
    print(`✅ Root user '${adminUser}' created successfully`);
} catch (e) {
    if (e.code === 51003) {
        print(`ℹ️  User '${adminUser}' already exists`);
    } else {
        print(`❌ Error creating user: ${e.message}`);
        throw e;
    }
}

// Switch to clinicai database (will create it if it doesn't exist)
db = db.getSiblingDB(dbName);

// Create a test collection to ensure database is created
db.createCollection('_init');

print(`✅ Database "${dbName}" created and ready`);

// Create indexes for common collections (optional - will be created by Beanie)
// db.patients.createIndex({ "patient_id": 1 }, { unique: true });
// db.visits.createIndex({ "visit_id": 1 }, { unique: true });

print('==================================================');
print('✅ MongoDB initialization complete!');
print('==================================================');

