-- MySQL dump 10.13  Distrib 5.7.24, for Linux (x86_64)
--
-- Host: localhost    Database: iris_dataset
-- ------------------------------------------------------
-- Server version	5.7.24-0ubuntu0.16.04.1

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `Iris`
--

DROP TABLE IF EXISTS `Iris`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `Iris` (
  `No` int(11) NOT NULL,
  `sepal_length` float DEFAULT NULL,
  `sepal_width` float DEFAULT NULL,
  `petal_length` float DEFAULT NULL,
  `petal_width` float DEFAULT NULL,
  `species` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`No`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Iris`
--

LOCK TABLES `Iris` WRITE;
/*!40000 ALTER TABLE `Iris` DISABLE KEYS */;
INSERT INTO `Iris` VALUES (1,5.1,3.5,1.4,0.2,'Iris-setosa'),(2,4.9,3,1.4,0.2,'Iris-setosa'),(3,4.7,3.2,1.3,0.2,'Iris-setosa'),(4,4.6,3.1,1.5,0.2,'Iris-setosa'),(5,5,3.6,1.4,0.2,'Iris-setosa'),(6,5.4,3.9,1.7,0.4,'Iris-setosa'),(7,4.6,3.4,1.4,0.3,'Iris-setosa'),(8,5,3.4,1.5,0.2,'Iris-setosa'),(9,4.4,2.9,1.4,0.2,'Iris-setosa'),(10,4.9,3.1,1.5,0.1,'Iris-setosa'),(11,5.4,3.7,1.5,0.2,'Iris-setosa'),(12,4.8,3.4,1.6,0.2,'Iris-setosa'),(13,4.8,3,1.4,0.1,'Iris-setosa'),(14,4.3,3,1.1,0.1,'Iris-setosa'),(15,5.8,4,1.2,0.2,'Iris-setosa'),(16,5.7,4.4,1.5,0.4,'Iris-setosa'),(17,5.4,3.9,1.3,0.4,'Iris-setosa'),(18,5.1,3.5,1.4,0.3,'Iris-setosa'),(19,5.7,3.8,1.7,0.3,'Iris-setosa'),(20,5.1,3.8,1.5,0.3,'Iris-setosa'),(21,5.4,3.4,1.7,0.2,'Iris-setosa'),(22,5.1,3.7,1.5,0.4,'Iris-setosa'),(23,4.6,3.6,1,0.2,'Iris-setosa'),(24,5.1,3.3,1.7,0.5,'Iris-setosa'),(25,4.8,3.4,1.9,0.2,'Iris-setosa'),(26,5,3,1.6,0.2,'Iris-setosa'),(27,5,3.4,1.6,0.4,'Iris-setosa'),(28,5.2,3.5,1.5,0.2,'Iris-setosa'),(29,5.2,3.4,1.4,0.2,'Iris-setosa'),(30,4.7,3.2,1.6,0.2,'Iris-setosa'),(31,4.8,3.1,1.6,0.2,'Iris-setosa'),(32,5.4,3.4,1.5,0.4,'Iris-setosa'),(33,5.2,4.1,1.5,0.1,'Iris-setosa'),(34,5.5,4.2,1.4,0.2,'Iris-setosa'),(35,4.9,3.1,1.5,0.1,'Iris-setosa'),(36,5,3.2,1.2,0.2,'Iris-setosa'),(37,5.5,3.5,1.3,0.2,'Iris-setosa'),(38,4.9,3.1,1.5,0.1,'Iris-setosa'),(39,4.4,3,1.3,0.2,'Iris-setosa'),(40,5.1,3.4,1.5,0.2,'Iris-setosa'),(41,5,3.5,1.3,0.3,'Iris-setosa'),(42,4.5,2.3,1.3,0.3,'Iris-setosa'),(43,4.4,3.2,1.3,0.2,'Iris-setosa'),(44,5,3.5,1.6,0.6,'Iris-setosa'),(45,5.1,3.8,1.9,0.4,'Iris-setosa'),(46,4.8,3,1.4,0.3,'Iris-setosa'),(47,5.1,3.8,1.6,0.2,'Iris-setosa'),(48,4.6,3.2,1.4,0.2,'Iris-setosa'),(49,5.3,3.7,1.5,0.2,'Iris-setosa'),(50,5,3.3,1.4,0.2,'Iris-setosa'),(51,7,3.2,4.7,1.4,'Iris-versicolor'),(52,6.4,3.2,4.5,1.5,'Iris-versicolor'),(53,6.9,3.1,4.9,1.5,'Iris-versicolor'),(54,5.5,2.3,4,1.3,'Iris-versicolor'),(55,6.5,2.8,4.6,1.5,'Iris-versicolor'),(56,5.7,2.8,4.5,1.3,'Iris-versicolor'),(57,6.3,3.3,4.7,1.6,'Iris-versicolor'),(58,4.9,2.4,3.3,1,'Iris-versicolor'),(59,6.6,2.9,4.6,1.3,'Iris-versicolor'),(60,5.2,2.7,3.9,1.4,'Iris-versicolor'),(61,5,2,3.5,1,'Iris-versicolor'),(62,5.9,3,4.2,1.5,'Iris-versicolor'),(63,6,2.2,4,1,'Iris-versicolor'),(64,6.1,2.9,4.7,1.4,'Iris-versicolor'),(65,5.6,2.9,3.6,1.3,'Iris-versicolor'),(66,6.7,3.1,4.4,1.4,'Iris-versicolor'),(67,5.6,3,4.5,1.5,'Iris-versicolor'),(68,5.8,2.7,4.1,1,'Iris-versicolor'),(69,6.2,2.2,4.5,1.5,'Iris-versicolor'),(70,5.6,2.5,3.9,1.1,'Iris-versicolor'),(71,5.9,3.2,4.8,1.8,'Iris-versicolor'),(72,6.1,2.8,4,1.3,'Iris-versicolor'),(73,6.3,2.5,4.9,1.5,'Iris-versicolor'),(74,6.1,2.8,4.7,1.2,'Iris-versicolor'),(75,6.4,2.9,4.3,1.3,'Iris-versicolor'),(76,6.6,3,4.4,1.4,'Iris-versicolor'),(77,6.8,2.8,4.8,1.4,'Iris-versicolor'),(78,6.7,3,5,1.7,'Iris-versicolor'),(79,6,2.9,4.5,1.5,'Iris-versicolor'),(80,5.7,2.6,3.5,1,'Iris-versicolor'),(81,5.5,2.4,3.8,1.1,'Iris-versicolor'),(82,5.5,2.4,3.7,1,'Iris-versicolor'),(83,5.8,2.7,3.9,1.2,'Iris-versicolor'),(84,6,2.7,5.1,1.6,'Iris-versicolor'),(85,5.4,3,4.5,1.5,'Iris-versicolor'),(86,6,3.4,4.5,1.6,'Iris-versicolor'),(87,6.7,3.1,4.7,1.5,'Iris-versicolor'),(88,6.3,2.3,4.4,1.3,'Iris-versicolor'),(89,5.6,3,4.1,1.3,'Iris-versicolor'),(90,5.5,2.5,4,1.3,'Iris-versicolor'),(91,5.5,2.6,4.4,1.2,'Iris-versicolor'),(92,6.1,3,4.6,1.4,'Iris-versicolor'),(93,5.8,2.6,4,1.2,'Iris-versicolor'),(94,5,2.3,3.3,1,'Iris-versicolor'),(95,5.6,2.7,4.2,1.3,'Iris-versicolor'),(96,5.7,3,4.2,1.2,'Iris-versicolor'),(97,5.7,2.9,4.2,1.3,'Iris-versicolor'),(98,6.2,2.9,4.3,1.3,'Iris-versicolor'),(99,5.1,2.5,3,1.1,'Iris-versicolor'),(100,5.7,2.8,4.1,1.3,'Iris-versicolor'),(101,6.3,3.3,6,2.5,'Iris-virginica'),(102,5.8,2.7,5.1,1.9,'Iris-virginica'),(103,7.1,3,5.9,2.1,'Iris-virginica'),(104,6.3,2.9,5.6,1.8,'Iris-virginica'),(105,6.5,3,5.8,2.2,'Iris-virginica'),(106,7.6,3,6.6,2.1,'Iris-virginica'),(107,4.9,2.5,4.5,1.7,'Iris-virginica'),(108,7.3,2.9,6.3,1.8,'Iris-virginica'),(109,6.7,2.5,5.8,1.8,'Iris-virginica'),(110,7.2,3.6,6.1,2.5,'Iris-virginica'),(111,6.5,3.2,5.1,2,'Iris-virginica'),(112,6.4,2.7,5.3,1.9,'Iris-virginica'),(113,6.8,3,5.5,2.1,'Iris-virginica'),(114,5.7,2.5,5,2,'Iris-virginica'),(115,5.8,2.8,5.1,2.4,'Iris-virginica'),(116,6.4,3.2,5.3,2.3,'Iris-virginica'),(117,6.5,3,5.5,1.8,'Iris-virginica'),(118,7.7,3.8,6.7,2.2,'Iris-virginica'),(119,7.7,2.6,6.9,2.3,'Iris-virginica'),(120,6,2.2,5,1.5,'Iris-virginica'),(121,6.9,3.2,5.7,2.3,'Iris-virginica'),(122,5.6,2.8,4.9,2,'Iris-virginica'),(123,7.7,2.8,6.7,2,'Iris-virginica'),(124,6.3,2.7,4.9,1.8,'Iris-virginica'),(125,6.7,3.3,5.7,2.1,'Iris-virginica'),(126,7.2,3.2,6,1.8,'Iris-virginica'),(127,6.2,2.8,4.8,1.8,'Iris-virginica'),(128,6.1,3,4.9,1.8,'Iris-virginica'),(129,6.4,2.8,5.6,2.1,'Iris-virginica'),(130,7.2,3,5.8,1.6,'Iris-virginica'),(131,7.4,2.8,6.1,1.9,'Iris-virginica'),(132,7.9,3.8,6.4,2,'Iris-virginica'),(133,6.4,2.8,5.6,2.2,'Iris-virginica'),(134,6.3,2.8,5.1,1.5,'Iris-virginica'),(135,6.1,2.6,5.6,1.4,'Iris-virginica'),(136,7.7,3,6.1,2.3,'Iris-virginica'),(137,6.3,3.4,5.6,2.4,'Iris-virginica'),(138,6.4,3.1,5.5,1.8,'Iris-virginica'),(139,6,3,4.8,1.8,'Iris-virginica'),(140,6.9,3.1,5.4,2.1,'Iris-virginica'),(141,6.7,3.1,5.6,2.4,'Iris-virginica'),(142,6.9,3.1,5.1,2.3,'Iris-virginica'),(143,5.8,2.7,5.1,1.9,'Iris-virginica'),(144,6.8,3.2,5.9,2.3,'Iris-virginica'),(145,6.7,3.3,5.7,2.5,'Iris-virginica'),(146,6.7,3,5.2,2.3,'Iris-virginica'),(147,6.3,2.5,5,1.9,'Iris-virginica'),(148,6.5,3,5.2,2,'Iris-virginica'),(149,6.2,3.4,5.4,2.3,'Iris-virginica'),(150,5.9,3,5.1,1.8,'Iris-virginica');
/*!40000 ALTER TABLE `Iris` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2018-10-28 14:33:20
