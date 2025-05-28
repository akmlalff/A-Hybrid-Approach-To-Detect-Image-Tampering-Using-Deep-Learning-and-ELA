const sqlite3 = require('sqlite3').verbose();
const bcrypt = require('bcryptjs');
const path = require('path');
const fs = require('fs');

let db; // <-- define globally

// Use the userData path provided by Electron
function initDatabase(userDataPath) {
    const dataDir = path.join(userDataPath, 'data');
    if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir, { recursive: true });
    }
    const dbPath = path.join(dataDir, 'users.db');
    db = new sqlite3.Database(dbPath);

    // Initialize database
    return new Promise((resolve, reject) => {
        db.serialize(() => {
            // First check if users table exists and has is_admin column
            db.get("PRAGMA table_info(users)", (err, row) => {
                if (err) {
                    console.error('Error checking table info:', err);
                    reject(err);
                    return;
                }

                // Check if users table exists
                db.get("SELECT name FROM sqlite_master WHERE type='table' AND name='users'", (err, tableExists) => {
                    if (err) {
                        console.error('Error checking if table exists:', err);
                        reject(err);
                        return;
                    }

                    if (!tableExists) {
                        // Create new users table with all required columns
                        db.run(`
                            CREATE TABLE users (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                email TEXT UNIQUE NOT NULL,
                                username TEXT UNIQUE NOT NULL,
                                password TEXT NOT NULL,
                                is_admin INTEGER DEFAULT 0,
                                last_login TEXT
                            )
                        `, (err) => {
                            if (err) {
                                console.error('Error creating users table:', err);
                                reject(err);
                                return;
                            }
                            checkAndCreateAdminUser(resolve, reject, userDataPath);
                        });
                    } else {
                        // Table exists, check if is_admin column exists
                        db.all("PRAGMA table_info(users)", (err, columns) => {
                            if (err) {
                                console.error('Error getting column info:', err);
                                reject(err);
                                return;
                            }

                            const hasIsAdminColumn = columns.some(col => col.name === 'is_admin');
                            
                            if (!hasIsAdminColumn) {
                                // Add is_admin column
                                db.run("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0", (err) => {
                                    if (err) {
                                        console.error('Error adding is_admin column:', err);
                                        reject(err);
                                        return;
                                    }
                                    console.log('Added is_admin column to users table');
                                    checkAndCreateAdminUser(resolve, reject, userDataPath);
                                });
                            } else {
                                checkAndCreateAdminUser(resolve, reject, userDataPath);
                            }
                        });
                    }
                });
            });
        });
    });
}

// Helper function to check and create admin user
function checkAndCreateAdminUser(resolve, reject, userDataPath) {
    const dataDir = path.join(userDataPath, 'data');
    const dbPath = path.join(dataDir, 'users.db');
    const db = new sqlite3.Database(dbPath);

    db.get('SELECT * FROM users WHERE username = ?', ['.'], (err, row) => {
        if (err) {
            console.error('Error checking admin user:', err);
            reject(err);
            return;
        }
        
        if (!row) {
            const hashedPassword = bcrypt.hashSync('.', 10);
            db.run('INSERT INTO users (email, username, password, is_admin) VALUES (?, ?, ?, ?)', 
                ['admin@system.com', 'admin', hashedPassword, 1], 
                (err) => {
                    if (err) {
                        console.error('Error creating admin user:', err);
                        reject(err);
                        return;
                    }
                    console.log('Admin user created successfully');
                    resolve();
                }
            );
        } else {
            // Ensure existing admin user has admin privileges
            db.run('UPDATE users SET is_admin = ? WHERE username = ?', [1, 'admin'], (err) => {
                if (err) {
                    console.error('Error updating admin privileges:', err);
                    reject(err);
                    return;
                }
                console.log('Admin privileges confirmed');
                resolve();
            });
        }
    });
}

// User registration
function registerUser(email, username, password) {
    return new Promise((resolve, reject) => {
        // Hash the password before storing
        const hashedPassword = bcrypt.hashSync(password, 10);
        
        db.run(
            'INSERT INTO users (email, username, password) VALUES (?, ?, ?)',
            [email, username, hashedPassword],
            function(err) {
                if (err) {
                    // Check for unique constraint violation
                    if (err.message.includes('UNIQUE constraint failed')) {
                        if (err.message.includes('users.email')) {
                            reject(new Error('Email already exists'));
                        } else if (err.message.includes('users.username')) {
                            reject(new Error('Username already exists'));
                        } else {
                            reject(err);
                        }
                    } else {
                        reject(err);
                    }
                    return;
                }
                
                resolve({ id: this.lastID, email, username });
            }
        );
    });
}

// User login
function loginUser(usernameOrEmail, password) {
    return new Promise((resolve, reject) => {
        // Check if input is email or username
        const query = 'SELECT * FROM users WHERE email = ? OR username = ?';
        
        db.get(query, [usernameOrEmail, usernameOrEmail], (err, user) => {
            if (err) {
                reject(err);
                return;
            }
            
            if (!user) {
                reject(new Error('User not found'));
                return;
            }
            
            // Compare password
            const isValid = bcrypt.compareSync(password, user.password);
            if (!isValid) {
                reject(new Error('Invalid password'));
                return;
            }
            
            // Update last login time
            const now = new Date().toISOString();
            db.run('UPDATE users SET last_login = ? WHERE id = ?', [now, user.id]);
            
            // Don't send password back
            const { password: _, ...userWithoutPassword } = user;
            resolve(userWithoutPassword);
        });
    });
}

// Get all users for admin dashboard
function getAllUsers() {
    return new Promise((resolve, reject) => {
        db.all('SELECT id, email, username, password, is_admin, last_login FROM users', (err, users) => {
            if (err) {
                reject(err);
                return;
            }
            resolve(users);
        });
    });
}

module.exports = {
    initDatabase,
    registerUser,
    loginUser,
    getAllUsers
}; 