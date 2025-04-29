# *Creating a Practical Component*

## **Database Creation**  
The database `Student-Society` is created to store student enrollment details in various societies.  

```sql
CREATE DATABASE Student_Society;
```
> Note: MySQL does not allow hyphens (`-`) in database names, so it is replaced with an underscore (`_`) or removed.  

Switch to the created database:  

```sql
USE Student_Society;
```

---

## **Table Structures**  

### **1. STUDENT Table**  
Stores student-related details like roll number, name, course, and date of birth.
```sql
CREATE TABLE STUDENT (
    RollNo CHAR(6) PRIMARY KEY,
    StudentName VARCHAR(20) NOT NULL,
    Course VARCHAR(10) NOT NULL,
    DOB DATE NOT NULL
    );
```

- `RollNo`: Unique identifier for each student (e.g., `S1001`, `X1002`).  
- `StudentName`: Full name of the student.  
- `Course`: Course in which the student is enrolled (e.g., `Computer Science`, `Biotechnology`).  
- `DOB`: Date of birth of the student.  

---

### **2. SOCIETY Table**  
Stores information about societies in the institution.  

```sql
CREATE TABLE SOCIETY (
    SocID CHAR(6) PRIMARY KEY,
    SocName VARCHAR(20) NOT NULL,
    MentorName VARCHAR(15) NOT NULL,
    TotalSeats INT UNSIGNED NOT NULL
    );
```

- `SocID`: Unique identifier for each society (e.g., `SOC001`).  
- `SocName`: Name of the society (e.g., `NCC`, `Debating Club`).  
- `MentorName`: Faculty mentor overseeing the society.  
- `TotalSeats`: Maximum capacity of the society. 

---

### **3. ENROLLMENT Table**  
Maintains the relationship between students and societies, storing their enrollment date.  

```sql
CREATE TABLE ENROLLMENT (
    RollNo CHAR(6) NOT NULL,
    SID CHAR(6) NOT NULL,
    DateOfEnrollment DATE,
    PRIMARY KEY (RollNo, SID),
    FOREIGN KEY (RollNo) REFERENCES STUDENT(RollNo) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (SID) REFERENCES SOCIETY(SocID) ON DELETE CASCADE ON UPDATE CASCADE
    );
```

- `RollNo`: Foreign key referencing `STUDENT(RollNo)`, ensuring every enrollment links to a valid student.  
- `SID`: Foreign key referencing `SOCIETY(SocID)`, ensuring enrollment is associated with an existing society.  
- `DateOfEnrollment`: The date on which the student joined the society.  
- `ON DELETE CASCADE`: If a student or society is removed, related enrollments are automatically deleted.  
- `ON UPDATE CASCADE`: Ensures that if a student's RollNo or a Society's ID changes, it is updated in all related records.  

---

## **Data Insertion**  

### **1. Insert Student Data** 
```sql
INSERT INTO STUDENT (RollNo, StudentName, Course, DOB) VALUES
('S1001', 'Ishika Purakayastha', 'Statistics', '1999-01-15'),
('S1002', 'Arun Kumar', 'Statistics', '1999-03-22'),
('S1003', 'Nishant Singh', 'Chemistry', '1999-05-10'),
('S1004', 'Renuka', 'Chemistry', '1999-07-08'),
('S1005', 'Geetanshu', 'Comp Sci', '2000-11-27'),
('S1006', 'Kohana Bhalla', 'Comp Sci', '2001-01-19'),
('S1007', 'Ratnesh Kumar', 'Physics', '2003-01-28'),
('S1008', 'Ananya Sharma', 'Physics', '2005-03-12'),
('S1009', 'Sourabh Kumar', 'Maths', '2007-09-28'),
('S1010', 'Vikash Kumar', 'Maths', '2008-10-14');
```
> More students can be added similarly.

---
### **2. Insert Society Data** 
```sql
INSERT INTO SOCIETY (SocID, SocName, Incharge, Capacity) VALUES
('S1', 'NCC', 'Dr. Sharma', 20),
('S2', 'NSS', 'Dr. Gupta', 15),
('S3', 'Debating', 'Mr. Mehta', 10),
('S4', 'Dancing', 'Ms. Singh', 20),
('S5', 'Sashakt', 'Dr. Yadav', 10),
('S6', 'Tech Club', 'Ms. Verma', 20),
('S7', 'Photography', 'Ms. Kapoor', 5);
```
> This ensures multiple societies are available for students.

---

### **3. Insert Enrollment Data**  
```sql
INSERT INTO ENROLLMENT (RollNo, SocID, JoinDate) VALUES
('S1001', 'S1', '2023-08-21'),
('S1001', 'S6', '2023-08-25'),
('S1002', 'S2', '2023-09-01'),
('S1003', 'S3', '2023-09-10'),
('S1004', 'S3', '2023-10-14'),
('S1005', 'S6', '2023-11-01'),
('S1006', 'S1', '2023-11-20'),
('S1007', 'S5', '2023-12-02'),
('S1008', 'S4', '2024-01-10'),
('S1009', 'S7', '2024-02-05');
```
> Each entry represents a student's membership in a society.

---